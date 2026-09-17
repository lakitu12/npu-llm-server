#!/usr/bin/env python3
"""Spark-X2.5 LiteRT-LM -> MT6991 NPU AOT 转换管线（CI 可复现版）.

流程: unpack -> 图改写(标量RESHAPE->SQUEEZE, 可选FC逐张量+行scale) -> host AOT
编译(MT6991/Neuron v8) -> 重打包并逐字节round-trip验证 -> report.json.

退出码:
  0 = 产出最终 .litertlm 且 round-trip 验证通过
  2 = blocked: 有改写/编译/DLA证据, 但无可用最终bundle
  1 = 错误 (含预期assert失败)
禁止: 把外层退出码或DLA存在当成模型成功.
"""
import argparse
import copy
import hashlib
import json
import mmap
import os
import pathlib
import re
import shutil
import struct
import sys
import time

import numpy as np


def log(*a):
    print(*a, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_model(path, s):
    f = path.open('rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return s.Model.GetRootAsModel(mm, 0), mm


def inventory(root, s, path):
    subs = []
    for gi in range(root.SubgraphsLength()):
        sg = root.Subgraphs(gi)
        name = sg.Name().decode('utf-8', 'replace') if sg.Name() else None
        subs.append({'index': gi, 'name': name, 'ops': sg.OperatorsLength()})
    return {'bytes': path.stat().st_size, 'buffers': root.BuffersLength(),
            'subgraphs': subs}


def apply_squeeze(mt, s):
    """标量目标 RESHAPE -> SQUEEZE (本机验证: 消除 host Neuron SIGSEGV)."""
    code = s.OperatorCodeT()
    code.builtinCode = s.BuiltinOperator.SQUEEZE
    code.deprecatedBuiltinCode = s.BuiltinOperator.SQUEEZE
    code.version = 1
    idx = len(mt.operatorCodes)
    mt.operatorCodes.append(code)
    changes, skipped = [], []
    for gi, g in enumerate(mt.subgraphs):
        for oi, o in enumerate(g.operators):
            if mt.operatorCodes[o.opcodeIndex].builtinCode != s.BuiltinOperator.RESHAPE:
                continue
            output = g.tensors[o.outputs[0]]
            if output.shape is not None and len(output.shape) != 0:
                continue
            if len(o.inputs) < 2 or list(g.tensors[o.inputs[0]].shape) != [1] \
                    or list(g.tensors[o.inputs[1]].shape) != [0]:
                skipped.append([gi, oi])
                continue
            o.opcodeIndex = idx
            o.inputs = np.array([o.inputs[0]], dtype=np.int32)
            o.builtinOptionsType = s.BuiltinOptions.SqueezeOptions
            o.builtinOptions = s.SqueezeOptionsT()
            o.builtinOptions.squeezeDims = [0]
            changes.append([gi, oi])
    return changes, skipped


def apply_fc_pertensor(mt, s):
    """逐通道scale INT8权重FC -> 单位scale FC + 行scale MUL (factor_spark_safe 忠实移植)."""
    mulcode = s.OperatorCodeT()
    mulcode.builtinCode = s.BuiltinOperator.MUL
    mulcode.deprecatedBuiltinCode = s.BuiltinOperator.MUL
    mulcode.version = 1
    mulidx = len(mt.operatorCodes)
    mt.operatorCodes.append(mulcode)
    changes, scale_buffers = [], {}
    for gi, g in enumerate(mt.subgraphs):
        newops = []
        for oi, o in enumerate(g.operators):
            if mt.operatorCodes[o.opcodeIndex].builtinCode != s.BuiltinOperator.FULLY_CONNECTED:
                newops.append(o)
                continue
            assert len(o.inputs) == 3 and o.inputs[2] == -1, f'FC@{gi}:{oi} 输入结构不符(应无bias)'
            assert o.builtinOptions is not None and o.builtinOptions.fusedActivationFunction == 0, \
                f'FC@{gi}:{oi} 带fused act'
            w = g.tensors[o.inputs[1]]
            output = g.tensors[o.outputs[0]]
            assert w.type == s.TensorType.INT8 and output.type == s.TensorType.FLOAT32, \
                f'FC@{gi}:{oi} dtype不符'
            q = w.quantization
            assert q is not None and q.quantizedDimension == 0 and len(q.scale) == w.shape[0] \
                and not np.any(q.zeroPoint), f'FC@{gi}:{oi} 量化布局不符'
            assert np.all(np.isfinite(q.scale)) and np.all(q.scale > 0), f'FC@{gi}:{oi} scale非法'
            key = (w.buffer, tuple(np.asarray(q.scale)))
            if key not in scale_buffers:
                buf = s.BufferT()
                buf.data = np.asarray(q.scale, dtype='<f4').view(np.uint8)
                mt.buffers.append(buf)
                scale_buffers[key] = len(mt.buffers) - 1
            sw = copy.deepcopy(w)
            sw.name = (w.name or b'weight') + b'_fc_unit_scale'
            sw.quantization.scale = np.array([1.0], np.float32)
            sw.quantization.zeroPoint = np.array([0], np.int64)
            g.tensors.append(sw)
            wi = len(g.tensors) - 1
            scale = s.TensorT()
            scale.name = b'fc_row_scales'
            scale.type = s.TensorType.FLOAT32
            scale.shape = [w.shape[0]]
            scale.hasRank = True
            scale.buffer = scale_buffers[key]
            g.tensors.append(scale)
            si = len(g.tensors) - 1
            tmp = copy.deepcopy(output)
            tmp.name = (output.name or b'output') + b'_before_row_scale'
            tmp.buffer = 0
            g.tensors.append(tmp)
            ti = len(g.tensors) - 1
            original_output = int(o.outputs[0])
            o.inputs = np.array([o.inputs[0], wi, -1], np.int32)
            o.outputs = np.array([ti], np.int32)
            mul = s.OperatorT()
            mul.opcodeIndex = mulidx
            mul.inputs = [ti, si]
            mul.outputs = [original_output]
            mul.builtinOptionsType = s.BuiltinOptions.MulOptions
            mul.builtinOptions = s.MulOptionsT()
            newops.extend([o, mul])
            changes.append([gi, oi, original_output])
        g.operators = newops
    return changes, scale_buffers


def reserialize_with_verify(mt, s, orig_root, orig_mm, dst):
    """元数据两遍打包 + 外部权重整体平移 + 逐字节回读验证."""
    import flatbuffers

    def build():
        b = flatbuffers.Builder(4 * 1024 * 1024)
        b.Finish(mt.Pack(b), file_identifier=b'TFL3')
        return bytes(b.Output())

    md = build()
    start = (len(md) + 4095) // 4096 * 4096
    for buf in mt.buffers:
        if buf.offset:
            buf.offset += start
    md = build()
    assert len(md) <= start, 'metadata grew past aligned header'
    root_off = struct.unpack_from('<I', md)[0]
    with dst.open('wb') as w:
        w.write(md)
        w.seek(start)
        for pos in range(0, len(orig_mm), 1 << 20):
            w.write(orig_mm[pos:pos + (1 << 20)])
    f2 = dst.open('rb')
    mm2 = mmap.mmap(f2.fileno(), 0, access=mmap.ACCESS_READ)
    new = s.Model.GetRootAsModel(mm2, 0)
    result = {'source_bytes': len(orig_mm), 'new_bytes': start + len(orig_mm),
              'metadata_bytes': len(md), 'root_offset': root_off}
    try:
        assert new.BuffersLength() >= orig_root.BuffersLength()
        for i in range(orig_root.BuffersLength()):
            a = orig_root.Buffers(i)
            z = new.Buffers(i)
            exp_off = a.Offset() + start if a.Offset() else 0
            assert (exp_off, a.Size(), a.DataLength()) == (z.Offset(), z.Size(), z.DataLength()), \
                f'buffer {i} 元数据不符'
            if z.Offset() and a.DataLength():
                assert bytes(a.DataAsNumpy()) == bytes(z.DataAsNumpy()), f'inline buffer {i} 内容不符'
        assert len(mm2) == start + len(orig_mm)
        for pos in range(0, len(orig_mm), 1 << 20):
            end = min(pos + (1 << 20), len(orig_mm))
            assert mm2[start + pos:start + end] == orig_mm[pos:end], f'payload drift at {pos}'
        result['external_payload_unchanged'] = True
        assert new.SubgraphsLength() == orig_root.SubgraphsLength()
        for gi in range(new.SubgraphsLength()):
            g = new.Subgraphs(gi)
            old = orig_root.Subgraphs(gi)
            for ti in range(old.TensorsLength()):
                a = old.Tensors(ti)
                z = g.Tensors(ti)
                assert a.Buffer() == z.Buffer() and a.Type() == z.Type(), \
                    f'原tensor {gi}:{ti} 位置/类型漂移'
        result['original_tensor_identity_preserved'] = True
    finally:
        mm2.close()
        f2.close()
    return result


def override_plugin(plugin_path):
    import ai_edge_litert
    d = pathlib.Path(ai_edge_litert.__file__).resolve().parent / \
        'vendors' / 'mediatek' / 'compiler' / 'libLiteRtCompilerPlugin_MediaTek.so'
    assert d.exists(), f'wheel plugin missing: {d}'
    shutil.copy2(plugin_path, d)
    return str(d)


def dissect_compiled(path, s, top=12):
    """编译产物 buffer 账本: 谁贡献了体积 (回答 'bundle 为什么这么大')."""
    root, mm = read_model(path, s)
    try:
        bufs = []
        for i in range(root.BuffersLength()):
            b = root.Buffers(i)
            bufs.append({'i': i, 'bytes': b.Size()})
        total = sum(x['bytes'] for x in bufs)
        big = sorted(bufs, key=lambda x: -x['bytes'])[:top]
        return {'file_bytes': path.stat().st_size, 'buffer_total': total,
                'num_buffers': len(bufs), 'top_buffers': big,
                'top_sum': sum(x['bytes'] for x in big)}
    finally:
        mm.close()


def compile_sections(tflite, work, subgraphs=None):
    import glob
    from ai_edge_litert.aot.aot_compile import aot_compile
    from ai_edge_litert.aot.vendors.mediatek.target import Target, SocModel, SocManufacturer
    outdir = work / 'aot' / tflite.stem
    if subgraphs:
        outdir = work / 'aot' / (tflite.stem + '_sg' + '-'.join(map(str, subgraphs)))
    outdir.mkdir(parents=True, exist_ok=True)
    dla = outdir / 'dla'
    dla.mkdir(parents=True, exist_ok=True)  # 插件要求目录已存在, 否则报 not a valid directory 且不落 DLA
    os.environ['MTKNN_ADAPTER_DLA_DIR'] = str(dla)
    entry = {'section': tflite.name, 'dla_dir': str(dla), 'target': 'MT6991(neuron v8)',
             'subgraphs_to_compile': subgraphs}
    t0 = time.time()
    err_before = set(glob.glob('/tmp/*.error'))
    try:
        aot_compile(str(tflite), output_dir=str(outdir),
                    target=Target(SocModel.MT6991, SocManufacturer.MEDIATEK),
                    backend_id='mediatek', keep_going=True,
                    subgraphs_to_compile=subgraphs)
        entry['compile_error'] = None
    except Exception as e:
        import traceback
        entry['compile_error'] = ''.join(
            traceback.format_exception(type(e), e, e.__traceback__))[-3000:]
    # apply_plugin(experimental_capture_stderr) 把插件真实报错写到 /tmp/<tmp>.error;
    # 抓取新增的, 否则 runner 一退证据就没了.
    new_errs = sorted(set(glob.glob('/tmp/*.error')) - err_before)
    blob = []
    for p in new_errs:
        try:
            blob.append(f'=== {os.path.basename(p)} ===\n' +
                        pathlib.Path(p).read_text(errors='replace')[-4000:])
        except OSError:
            pass
    if blob:
        entry['apply_plugin_stderr'] = '\n'.join(blob)[-8000:]
    entry['seconds'] = round(time.time() - t0, 1)
    entry['outputs'] = []
    for p in sorted(outdir.rglob('*.tflite')):
        if not p.is_file() or p == tflite:
            continue
        size = p.stat().st_size
        entry['outputs'].append({'name': p.name, 'bytes': size,
                                 'sha256': sha256_file(p) if size else None,
                                 'path': str(p)})
    if dla.exists():
        files = sorted(p for p in dla.rglob('*') if p.is_file())
        mani = {str(p.relative_to(dla)): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                for p in files[:400]}
        (outdir / 'dla_manifest.json').write_text(json.dumps(mani, indent=1))
        entry['dla_files'] = len(files)
        entry['dla_unique_sha'] = len({v['sha256'] for v in mani.values()}) if mani else 0
    log(f"[compile] {tflite.name} err={entry['compile_error'] is not None} "
        f"outputs={[(o['name'], o['bytes']) for o in entry['outputs']]} "
        f"dla_files={entry.get('dla_files')}")
    return [entry]


def repack_and_verify(unpack_dir, out_entry, out_dir, variant):
    out_dir.mkdir(parents=True, exist_ok=True)
    compiled = pathlib.Path(out_entry['path'])
    staged = unpack_dir / 'compiled_section.tflite'
    shutil.copy2(compiled, staged)
    toml = (unpack_dir / 'model.toml').read_text()
    new_toml, n = re.subn(r'data_path = "Section[0-9]+_TFLiteModel[^"]*\.tflite"',
                          'data_path = "compiled_section.tflite"', toml)
    if n != 1:
        return {'roundtrip_verified': False, 'reason': f'TFLiteModel data_path 匹配数={n}'}
    (unpack_dir / 'model.compiled.toml').write_text(new_toml)
    from litert_lm_builder import pack
    from litert_lm_builder import unpack as ltlm_unpack
    out = out_dir / f'Spark-X2.5-4B_{variant}_mt6991_npu.litertlm'
    pack(str(unpack_dir / 'model.compiled.toml'), str(out))
    vdir = out_dir / 'verify_unpack'
    if vdir.exists():
        shutil.rmtree(vdir)
    ltlm_unpack(str(out), str(vdir))
    # NOTE: unpack/peek 落盘名由 Section{N}_TFLiteModel_{model_type}.tflite 规则
    # 生成, 不沿用 toml 的 data_path (上轮 round-trip 假阴性根因)。
    cands = sorted(vdir.glob('Section*_TFLiteModel*.tflite'))
    info = {'sections_back': len(list(vdir.iterdir()))}
    ok = len(cands) == 1 and sha256_file(cands[0]) == out_entry['sha256']
    info['dumped_name'] = cands[0].name if len(cands) == 1 else [p.name for p in cands]
    return {'path': str(out), 'bytes': out.stat().st_size, 'sha256': sha256_file(out),
            'roundtrip_verified': bool(ok), 'roundtrip_detail': info,
            'source_compiled_sha256': out_entry['sha256']}


EXPECTED = {'int8': {'squeeze': 7, 'fc': 1483}}


def main():
    ap = argparse.ArgumentParser(description='Spark litertlm -> MT6991 NPU AOT (honest report)')
    ap.add_argument('--input', required=True)
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--variant', default='int8', choices=['int8', 'int4'])
    ap.add_argument('--rewrite', default='squeeze', choices=['squeeze', 'none'])
    ap.add_argument('--fc', action='store_true')
    ap.add_argument('--plugin', default=None)
    ap.add_argument('--skip-compile', action='store_true')
    ap.add_argument('--no-expect', action='store_true',
                    help='跳过 squeeze/FC 预期数量断言 (冒烟测试用)')
    ap.add_argument('--subgraphs', default=None,
                    help='逗号分隔子图索引(如 0): 只编这些子图压内存峰值; '
                         '产物为部分编译, 不做bundle重打包')
    args = ap.parse_args()

    work = pathlib.Path(args.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    rc = 1
    report = {
        'status': 'error',
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'variant': args.variant,
        'rewrite': {'squeeze': args.rewrite == 'squeeze', 'fc': args.fc},
        'plugin': 'wheel' if not args.plugin else 'custom',
        'python': sys.version.split()[0],
    }
    try:
        from ai_edge_litert import schema_py_generated as s
        from importlib import metadata
        for pkg in ('ai-edge-litert', 'ai-edge-litert-sdk-mediatek',
                    'litert-lm-builder', 'flatbuffers', 'numpy'):
            try:
                report.setdefault('versions', {})[pkg] = metadata.version(pkg)
            except Exception:
                pass
        if args.plugin:
            report['plugin_path'] = override_plugin(pathlib.Path(args.plugin).resolve())
        inp = pathlib.Path(args.input).resolve()
        report['input'] = {'path': str(inp), 'bytes': inp.stat().st_size,
                           'sha256': sha256_file(inp)}

        from litert_lm_builder import unpack as ltlm_unpack
        unpack_dir = work / 'unpack'
        if not (unpack_dir / 'model.toml').exists():
            log('[1/4] unpack ...')
            ltlm_unpack(str(inp), str(unpack_dir))
        tflites = sorted(p for p in unpack_dir.rglob('*.tflite') if p.is_file())
        report['unpack'] = {'tflites': [p.name for p in tflites]}
        assert len(tflites) == 1, f'期望1个tflite段, 实际: {[p.name for p in tflites]}'
        src = tflites[0]
        root, mm = read_model(src, s)
        report['model_inventory'] = inventory(root, s, src)

        mt = s.ModelT.InitFromObj(root)
        sq_changes, sq_skipped, scale_buffers = [], [], {}
        if args.rewrite == 'squeeze':
            log('[2/4] squeeze rewrite ...')
            sq_changes, sq_skipped = apply_squeeze(mt, s)
        fc_changes = []
        if args.fc:
            log('[2b/4] fc per-tensor rewrite ...')
            fc_changes, scale_buffers = apply_fc_pertensor(mt, s)

        exp = {} if args.no_expect else EXPECTED.get(args.variant, {})
        if args.rewrite == 'squeeze':
            if sq_skipped:
                log('WARN: 不符合改写模式的标量RESHAPE:', sq_skipped)
            if 'squeeze' in exp and len(sq_changes) != exp['squeeze']:
                raise AssertionError(
                    f"squeeze改写数 {len(sq_changes)} != 预期 {exp['squeeze']} (int8本机已验证值)")
        if args.fc and 'fc' in exp and len(fc_changes) != exp['fc']:
            raise AssertionError(
                f"FC改写数 {len(fc_changes)} != 预期 {exp['fc']} (int8本机已验证值)")

        if sq_changes or fc_changes:
            log('[3/4] reserialize + 逐字节验证 ...')
            rewritten = work / 'rewrite' / 'model.tflite'
            rewritten.parent.mkdir(exist_ok=True)
            validation = reserialize_with_verify(mt, s, root, mm, rewritten)
            validation.update({
                'squeeze_changes': sq_changes,
                'squeeze_skipped': sq_skipped,
                'fc_rewrites': len(fc_changes),
                'scale_buffers': len(scale_buffers),
                'micro_cpu_parity': 'proven locally only (convert-work/FINDINGS.md), not re-run in CI',
            })
            (rewritten.parent / 'validation.json').write_text(json.dumps(validation, indent=1))
            report['rewrite'] = validation
            model_for_compile = rewritten
            # 注意: 不关 mm — InitFromObj 的 DataAsNumpy 是零拷贝视图,
            # mt 仍引用 mm 缓冲; 进程退出时自动释放.
        else:
            model_for_compile = src

        if args.skip_compile:
            report['compile'] = []
        else:
            subs = [int(x) for x in args.subgraphs.split(',')] if args.subgraphs else None
            log(f'[4/4] AOT compile ... subgraphs={subs or "all"}')
            report['compile'] = compile_sections(model_for_compile, work, subgraphs=subs)

        cands = [o for e in report.get('compile', []) for o in e['outputs'] if o['bytes'] > 0]
        if cands:
            big = max(cands, key=lambda o: o['bytes'])
            try:
                report['compiled_dissection'] = dissect_compiled(
                    pathlib.Path(big['path']), s)
            except Exception as ex:
                report['compiled_dissection'] = {'error': str(ex)[:300]}
            bundle = repack_and_verify(unpack_dir, big, work / 'out', args.variant)
            if args.subgraphs:
                bundle['partial'] = True
                bundle['note'] = f'仅子图 {args.subgraphs} 编译; 整bundle其余部分仍为CPU版'
            report['bundle'] = bundle
            if bundle.get('roundtrip_verified'):
                report['status'] = 'compiled_bundle_verified'
                rc = 0
            else:
                report['status'] = 'blocked'
                report['blocked_reason'] = bundle.get('reason', 'bundle round-trip verify failed')
                rc = 2
        else:
            report['status'] = 'blocked'
            report['blocked_reason'] = '无>0字节编译产物 (见 compile 条目; 打包/内存墙未解除)'
            rc = 2
        report['evidence_note'] = ('DLA存在≠设备可加载模型; '
                                   '只有 compiled_bundle_verified 才算转换成功')
    except AssertionError as e:
        report['error'] = f'assert: {e}'
    except Exception:
        import traceback
        report['error'] = traceback.format_exc()[-4000:]
    finally:
        report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (work / 'report.json').write_text(json.dumps(report, indent=1))
        log('STATUS:', report['status'], '| error:', report.get('error'),
            '| blocked:', report.get('blocked_reason'))
    sys.exit(rc)


if __name__ == '__main__':
    main()
