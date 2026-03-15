// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DemucsMLX",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "DemucsMLX", targets: ["DemucsMLX"]),
        .executable(name: "demucs-separate", targets: ["Separator"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1"),
        .package(url: "https://github.com/apple/swift-argument-parser", "1.3.0"..<"1.6.0"),
    ],
    targets: [
        .target(
            name: "DemucsMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/DemucsMLX"
        ),
        .executableTarget(
            name: "Separator",
            dependencies: [
                "DemucsMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/Separator"
        ),
        .testTarget(
            name: "DemucsMLXTests",
            dependencies: ["DemucsMLX"],
            path: "Tests/DemucsMLXTests"
        ),
    ]
)
