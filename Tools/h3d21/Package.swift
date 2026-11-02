// swift-tools-version: 6.0
import PackageDescription

// Developer harness for the Hunyuan3D 2.1 port: parity vs PyTorch fixtures and end-to-end meshing.
// Kept out of Hy3DMLX so that package's scheme/tests are unchanged. The XCTest host cannot read the
// external models drive, a plain executable can.
let package = Package(
    name: "h3d21",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../../Packages/Hy3DMLX"),
        .package(path: "../../vendor/mlx-swift"),
    ],
    targets: [
        .executableTarget(name: "h3d21", dependencies: [
            .product(name: "Hy3DMLX", package: "Hy3DMLX"),
            .product(name: "MLX", package: "mlx-swift"),
        ]),
    ]
)
