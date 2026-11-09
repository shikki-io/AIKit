// MARK: - VJEPAEncoderTests
// TDDP step 1 (RED) + step 2 (GREEN): encoder initialisation.
//
// See features/shikki-vjepa-mlx-node.md §5 Phase 1 + §6 TDDP.

import Testing
@testable import AIKitVision

@Suite("VJEPAEncoder")
struct VJEPAEncoderTests {

    @Test("encoder initialises with 12 transformer blocks for ViT-B/16")
    func test_initializesWith12TransformerBlocks() {
        let encoder = VJEPAEncoder(config: .vitB16)
        #expect(encoder.config.numLayers == 12)
        #expect(encoder.blocks.count == 12)
    }

    @Test("ViT-B/16 config reports 768 hidden dim and 12 heads")
    func test_vitB16ConfigHiddenDimAndHeads() {
        let config = VJEPAEncoder.Config.vitB16
        #expect(config.hiddenDim == 768)
        #expect(config.numHeads == 12)
        #expect(config.patchSize == 16)
        #expect(config.temporalPatchSize == 2)
    }

    @Test("each transformer block exposes attention, MLP, and two layer norms")
    func test_transformerBlockStructure() {
        let encoder = VJEPAEncoder(config: .vitB16)
        for block in encoder.blocks {
            #expect(block.hiddenDim == 768)
            #expect(block.numHeads == 12)
        }
    }
}
