DERIVED    := .derived
XCODEBUILD := xcodebuild -project GrizzyModlr.xcodeproj -scheme GrizzyModlr \
              -derivedDataPath $(DERIVED)

.PHONY: run build release package test smoke gen clean

# Build (Debug) and launch the app
run: build
	open $(DERIVED)/Build/Products/Debug/GrizzyModlr.app

# Regenerate the Xcode project from project.yml
gen:
	xcodegen generate

build: gen
	$(XCODEBUILD) -configuration Debug CODE_SIGNING_ALLOWED=NO build

# Apple-silicon-only Release build (Float16 has no x86_64 slice)
release: gen
	$(XCODEBUILD) -configuration Release ARCHS=arm64 CODE_SIGNING_ALLOWED=YES build
	@echo "app: $(DERIVED)/Build/Products/Release/GrizzyModlr.app"

# Zip a signed Release app for distribution. `ditto` preserves the app bundle's
# metadata and resource forks, unlike a plain `zip` invocation.
package: release
	rm -rf dist
	mkdir -p dist
	ditto -c -k --sequesterRsrc --keepParent \
		$(DERIVED)/Build/Products/Release/GrizzyModlr.app dist/GrizzyModlr.zip

test: gen
	$(XCODEBUILD) CODE_SIGNING_ALLOWED=NO test

# Headless self-drive: onboarding -> import -> generate -> paint -> export -> screenshots
smoke: build
	MODELR_UI_SMOKE=1 MODELR_SMOKE_MODEL=small \
	MODELR_SMOKE_OUT=$(CURDIR)/docs/e2e/smoke-make \
	$(DERIVED)/Build/Products/Debug/GrizzyModlr.app/Contents/MacOS/GrizzyModlr

clean:
	rm -rf $(DERIVED)
