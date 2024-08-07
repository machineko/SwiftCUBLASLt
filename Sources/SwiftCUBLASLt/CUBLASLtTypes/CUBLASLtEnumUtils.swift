import cxxCUBLASLt
// import SwiftCUBLAS


public extension cublasLtOrder {
    var ascublasLt: cublasLtOrder_t {
        #if os(Linux)
            return .init(UInt32(self.rawValue))
        #elseif os(Windows)
            return .init(Int32(self.rawValue))
        #else
            fatalerror()
        #endif
    }
}