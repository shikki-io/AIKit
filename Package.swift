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
        // ShiNetKit v1.0.0 — module renamed at source (collision fix, no aliases ever).
        .package(url: "https://github.com/FJ-Studios/NetKit.git", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "AIKit",
            dependencies: [
                .product(name: "ShiNetKit", package: "NetKit"),
            ]
        ),
        .testTarget(
            name: "AIKitTests",
            dependencies: [
                "AIKit",
                .product(name: "ShiNetKit", package: "NetKit"),
            ]
        ),
    ]
)
