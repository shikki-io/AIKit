// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "AIKit",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "AIKit",
            targets: ["AIKit"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/FJ-Studios/NetKit.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "AIKit",
            dependencies: [
                .product(name: "NetKit", package: "NetKit"),
            ]
        ),
        .testTarget(
            name: "AIKitTests",
            dependencies: [
                "AIKit",
                .product(name: "NetKit", package: "NetKit"),
            ]
        ),
    ]
)
