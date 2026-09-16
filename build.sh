set -eo pipefail
# hand-build APK: aapt2 + javac + d8 + zipalign + apksigner. No gradle.
cd "$(dirname "$0")"
SDK=~/Android/Sdk
BT=$SDK/build-tools/36.1.0
PLATFORM=$SDK/platforms/android-36
OUT=out
rm -rf $OUT
mkdir -p $OUT/gen $OUT/classes assets

echo "== check assets =="
ls -lh assets/
ls -lh lib/arm64-v8a/

echo "== aapt2 compile =="
$BT/aapt2 compile --dir res -o $OUT/res.zip 2>&1 | head -5 || true
mkdir -p res/values
[ -f res/values/strings.xml ] || echo '<?xml version="1.0" encoding="utf-8"?><resources><string name="app_name">NPU LLM</string></resources>' > res/values/strings.xml
$BT/aapt2 compile --dir res -o $OUT/res.zip

echo "== aapt2 link =="
$BT/aapt2 link -o $OUT/base.apk -I $PLATFORM/android.jar \
  --manifest AndroidManifest.xml --java $OUT/gen $OUT/res.zip \
  --min-sdk-version 28 --target-sdk-version 34

echo "== javac =="
find src $OUT/gen -name "*.java" > $OUT/sources.txt
cat $OUT/sources.txt
javac -source 8 -target 8 -bootclasspath $PLATFORM/android.jar \
  -classpath $PLATFORM/android.jar -d $OUT/classes @${OUT}/sources.txt 2>&1 | grep -v "bootstrap\|warning" | head -20 || true

echo "== d8 =="
$BT/d8 --release --lib $PLATFORM/android.jar --min-api 28 \
  $(find $OUT/classes -name "*.class" | tr '\n' ' ') --output $OUT/ 2>&1 | head -10

echo "== add dex =="
cd $OUT && zip -q base.apk classes.dex && cd ..

echo "== add native libs (stored, lib/ABI path) =="
mkdir -p $OUT/apklib/lib/arm64-v8a
cp lib/arm64-v8a/*.so $OUT/apklib/lib/arm64-v8a/
cd $OUT/apklib && zip -0 -q -r ../base.apk lib/ && cd ../..
unzip -l $OUT/base.apk | grep -e "\.so" -e "dex"

echo "== add assets =="
$BT/aapt2 link -o /dev/null 2>/dev/null || true
python3 - <<'EOF'
import zipfile, os
apk = 'out/base.apk'
for f in sorted(os.listdir('assets')):
    src = os.path.join('assets', f)
    with zipfile.ZipFile(apk, 'a', zipfile.ZIP_STORED) as z:
        z.write(src, 'assets/' + f)
    print('added assets/' + f, os.path.getsize(src))
EOF

echo "== zipalign + sign =="
$BT/zipalign -f -p 4 $OUT/base.apk $OUT/aligned.apk
[ -f debug.keystore ] || keytool -genkeypair -keystore debug.keystore -alias dbg \
  -keyalg RSA -keysize 2048 -validity 3650 -storepass android -keypass android \
  -dname "CN=debug" 2>&1 | tail -1
$BT/apksigner sign --ks debug.keystore --ks-pass "pass:android" --key-pass "pass:android" \
  --out $OUT/npu-llm.apk $OUT/aligned.apk
$BT/apksigner verify $OUT/npu-llm.apk && echo SIGNED-OK
ls -lh $OUT/npu-llm.apk
