import SwiftCU
import SwiftCUBLAS
import cxxCUBLASLt

extension CUBLASLtHandle {
    public func gemm<inputType: CUBLASDataType, outputType: CUBLASDataType, computeType: CUBLASDataType>(
        desc: borrowing CUBLASLtMatMulDescriptor, params: inout CUBLASLtaramsMixed<inputType, outputType, computeType>
    ) -> cublasStatus {
        let status = cublasLtMatmul(
            handle, desc.desc, &params.alpha, params.A, params.aDesc.layout,
            params.B, params.bDesc.layout, &params.beta, params.C,
            params.cDesc.layout, params.C, params.cDesc.layout,
            nil, nil, 0, nil
        ).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't run cublasLtMatmul function \(status)")
        #endif
        return status
    }

    public func gemm<inputType: CUBLASDataType, outputType: CUBLASDataType, computeType: CUBLASDataType>(
        desc: inout CUBLASLtMatMulDescriptor, params: inout CUBLASLtaramsMixed<inputType, outputType, computeType>, stream: inout cudaStream
    ) -> cublasStatus {
        let status = cublasLtMatmul(
            handle, desc.desc, &params.alpha, params.A, params.aDesc.layout,
            params.B, params.bDesc.layout, &params.beta, params.C,
            params.cDesc.layout, params.C, params.cDesc.layout,
            nil, nil, 0, stream.stream
        ).asSwift
        #if safetyCheck
            status.safetyCheckCondition(message: "Can't run cublasLtMatmul function \(status)")
        #endif
        return status
    }
}
