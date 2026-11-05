#!/bin/bash
# Builds Forma.app (Apple silicon and Intel in one) and Forma.dmg into dist/.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
SHORT=${VERSION%%-*}
export MACOSX_DEPLOYMENT_TARGET=11.0
rustup target add aarch64-apple-darwin x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
cargo build --release --target x86_64-apple-darwin

rm -rf dist/Forma.app && mkdir -p dist/Forma.app/Contents/MacOS dist/Forma.app/Contents/Resources
lipo -create -output dist/Forma.app/Contents/MacOS/forma \
  target/aarch64-apple-darwin/release/forma target/x86_64-apple-darwin/release/forma
sed -e "s/@SHORT@/$SHORT/" -e "s/@VERSION@/$VERSION/" macos/Info.plist > dist/Forma.app/Contents/Info.plist
printf 'APPL????' > dist/Forma.app/Contents/PkgInfo

# icon from the logo
ICONSET=$(mktemp -d)/Forma.iconset; mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s assets/logo.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  d=$((s * 2)); sips -z $d $d assets/logo.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o dist/Forma.app/Contents/Resources/Forma.icns

# ad-hoc signature: required for Apple silicon, no developer account needed
codesign --force --deep --sign - dist/Forma.app
codesign --verify --verbose dist/Forma.app

# disk image with a shortcut to Applications
STAGE=$(mktemp -d)
cp -R dist/Forma.app "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f dist/Forma.dmg
hdiutil create -volname "Forma $VERSION" -srcfolder "$STAGE" -ov -format UDZO dist/Forma.dmg
ditto -c -k --keepParent dist/Forma.app dist/Forma-mac.zip
lipo -info dist/Forma.app/Contents/MacOS/forma
