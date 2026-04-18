import Testing
@testable import AIKit

@Suite("AICapabilities")
struct AICapabilitiesTests {

    @Test("Union combines capabilities")
    func union() {
        let a: AICapabilities = [.textGeneration, .translation]
        let b: AICapabilities = [.translation, .vision]
        let combined = a.union(b)
        #expect(combined.contains(.textGeneration))
        #expect(combined.contains(.translation))
        #expect(combined.contains(.vision))
    }

    @Test("Intersection finds common capabilities")
    func intersection() {
        let a: AICapabilities = [.textGeneration, .translation]
        let b: AICapabilities = [.translation, .vision]
        let common = a.intersection(b)
        #expect(common.contains(.translation))
        #expect(!common.contains(.textGeneration))
        #expect(!common.contains(.vision))
    }

    @Test("Contains checks single capability")
    func contains() {
        let caps: AICapabilities = [.textGeneration, .toolUse, .embedding]
        #expect(caps.contains(.textGeneration))
        #expect(caps.contains(.toolUse))
        #expect(caps.contains(.embedding))
        #expect(!caps.contains(.vision))
        #expect(!caps.contains(.ocr))
    }

    @Test("All ten capabilities are distinct")
    func allCapabilitiesDistinct() {
        let all: [AICapabilities] = [
            .textGeneration, .translation, .ocr, .imageGeneration,
            .voiceToText, .textToVoice, .inpainting, .embedding,
            .vision, .toolUse,
        ]
        for (i, a) in all.enumerated() {
            for (j, b) in all.enumerated() where i != j {
                #expect(!a.contains(b), "Capability \(i) should not contain \(j)")
            }
        }
    }

    @Test("Empty capabilities contains nothing")
    func empty() {
        let empty: AICapabilities = []
        #expect(!empty.contains(.textGeneration))
        #expect(empty.isEmpty)
    }
}
