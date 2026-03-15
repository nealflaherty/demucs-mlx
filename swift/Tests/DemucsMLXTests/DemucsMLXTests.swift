import XCTest
@testable import DemucsMLX

final class DemucsMLXTests: XCTestCase {
    func testVersion() {
        XCTAssertEqual(DemucsMLX.version, "0.1.0")
    }
}
