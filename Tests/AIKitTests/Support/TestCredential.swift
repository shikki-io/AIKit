// TestCredential.swift — the placeholder credentials the tests send.
//
// A test needs a value to prove an API key reaches the right header; it never
// needs one shaped like a real key. A provider-prefixed literal reads as a
// leaked secret to the secret scan and to a human reviewer alike, so every
// placeholder is spelled here, once, and is visibly not a key.

/// Placeholder credentials for tests. None of them is, or looks like, a real key.
enum TestCredential {
    /// The credential a test hands an OpenAI-shaped provider.
    static let openAI = "test-credential-openai"
    /// The credential a test hands an Anthropic-shaped provider.
    static let anthropic = "test-credential-anthropic"
    /// A second, distinct credential, for tests that compare two.
    static let other = "test-credential-other"
}
