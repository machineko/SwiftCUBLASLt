import SwiftCU
import SwiftCUBLAS
// import cxxCUBLASLt
import cxxCUBLASLt

extension cublasStatus_t {
    /// Converts the `cublasStatus_t` to a Swift `cublasStatus`.
    var asSwift: cublasStatus {
        return cublasStatus(rawValue: Int(self.rawValue))!
    }
}

/// A structure that manages a CUBLASLt handle.
public struct CUBLASLtHandle: ~Copyable {
    /// The underlying CUBLASLt handle.
    public var handle: cublasLtHandle_t?

    /// Initializes a new CUBLASLt handle.
    public init() {
        var handle: cublasHandle_t?
        let status = cublasLtCreate(&handle).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't create cublas handle cublasError: \(status)")
        #endif
        self.handle = handle
    }

    /// Deinitializes the CUBLASLt handle, releasing any associated resources.
    deinit {
        let status = cublasLtDestroy(handle).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't destroy cublas handle cublasError: \(status)")
        #endif
    }
}

/// A structure that holds parameters for CUBLAS operations with mixed data types.
public struct CUBLASLtaramsMixed<inputType: CUBLASDataType, outputType: CUBLASDataType, computeType: CUBLASDataType>: ~Copyable {
    /// Pointer to the first matrix (A).
    public var A: UnsafePointer<inputType>
    /// Pointer to the second matrix (B).
    public var B: UnsafePointer<inputType>
    /// Pointer to the result matrix (C).
    public var C: UnsafeMutablePointer<outputType>

    public var aDesc: CUBLASLtMatrixLayout
    public var bDesc: CUBLASLtMatrixLayout
    public var cDesc: CUBLASLtMatrixLayout

    /// Scalar multiplier for the product of matrices A and B.
    public var alpha: computeType
    /// Scalar multiplier for the matrix C.
    public var beta: computeType
}


public struct CUBLASLtMatrixLayout: ~Copyable {
    var layout: cublasLtMatrixLayout_t?

    public init(dataType: swiftCUDADataType, rows: UInt64, columns: UInt64, ld: Int64) {
        var layout: cublasLtMatrixLayout_t?
        let status = cublasLtMatrixLayoutCreate(&layout, dataType.asCUDA, rows, columns, ld).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't create cublasLt layout cublasError: \(status)")
        #endif
        self.layout = layout
    }

    public init(fromRowMajor dataType: swiftCUDADataType, rows: UInt64, columns: UInt64, ld: Int64) {
        var layout: cublasLtMatrixLayout_t?
        let status = cublasLtMatrixLayoutCreate(&layout, dataType.asCUDA, rows, columns, ld).asSwift
        var order = cublasLtOrder.cublaslt_order_row.ascublasLt
        let matrixLayoutStatus = cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
            MemoryLayout<cublasLtOrder_t>.size
        ).asSwift

        // #if safetyCheck
        status.safetyCheckCondition(message: "Can't create cublasLt layout")
        matrixLayoutStatus.safetyCheckCondition(message: "Can't set matrix layout")
        // #endif
        self.layout = layout
    }

    public init(fromRowMajor dataType: swiftCUDADataType, rows: Int, columns: Int, ld: Int) {
        var layout: cublasLtMatrixLayout_t?
        let status = cublasLtMatrixLayoutCreate(&layout, dataType.asCUDA, UInt64(rows), UInt64(columns), Int64(ld)).asSwift
        var order = cublasLtOrder.cublaslt_order_row.ascublasLt
        let matrixLayoutStatus = cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
            MemoryLayout<cublasLtOrder_t>.size
        ).asSwift

        #if safetyCheck
            status.safetyCheckCondition(message: "Can't create cublasLt layout")
            matrixLayoutStatus.safetyCheckCondition(message: "Can't set matrix layout")
        #endif
        self.layout = layout
    }

    deinit {
        let status = cublasLtMatrixLayoutDestroy(layout).asSwift
        #if safetyCheck
            precondition(status.isSuccessful, "Can't destroy cublasLt layout cublasError: \(status)")
        #endif
    }
}

public struct CUBLASLtMatMulDescriptor: ~Copyable {
    var desc: cublasLtMatmulDesc_t?

    public init(computeType: cublasComputeType, dataType: swiftCUDADataType) {
        var desc: cublasLtMatmulDesc_t?
        let status = cublasLtMatmulDescCreate(&desc, computeType.ascublas, dataType.asCUDA).asSwift
        status.safetyCheckCondition(message: "Can't create cublasLt matmul descriptor")
        self.desc = desc
    }

    deinit {
        let status = cublasLtMatmulDescDestroy(desc).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't destory cublasLt matmul descriptor")
        #endif
    }
}
