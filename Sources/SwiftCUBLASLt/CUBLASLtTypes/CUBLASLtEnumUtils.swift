import cxxCUBLASLt

// import SwiftCUBLAS

extension cublasLtOrder {
    public var ascublasLt: cublasLtOrder_t {
        #if os(Linux)
            return .init(UInt32(self.rawValue))
        #elseif os(Windows)
            return .init(Int32(self.rawValue))
        #else
            fatalerror()
        #endif
    }
}
