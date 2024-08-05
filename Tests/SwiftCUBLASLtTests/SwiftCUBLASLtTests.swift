import XCTest
@testable import SwiftCUBLASLt
import cxxCUBLASLt
final class SwiftCUBLASLtTests: XCTestCase {
    func testExample() throws {
        var handle: cublasLtHandle_t?
        let result = cublasLtCreate(&handle)
        print(result)
        XCTAssert(result == .init(0))
    }
}
