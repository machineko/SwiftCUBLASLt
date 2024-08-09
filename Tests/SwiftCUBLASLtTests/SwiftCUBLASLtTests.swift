import SwiftCU
import SwiftCUBLAS
import XCTest
import cxxCUBLASLt

@testable import SwiftCUBLASLt

final class SwiftCUBLASLtTests: XCTestCase {
    func testMatmulF16() throws {
        let handle = CUBLASLtHandle()
        _ = CUDADevice(index: 0).setDevice()
        let m = 2
        let n = 2
        let k = 4

        var A: [Float16] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]

        var B: [Float16] = [
            8.0, 7.0,
            6.0, 5.0,
            4.0, 3.0,
            2.0, 1.0,
        ]

        var C: [Float16] = [Float16](repeating: 0.0, count: m * n)

        var aPointer: UnsafeMutableRawPointer?
        var bPointer: UnsafeMutableRawPointer?
        var cPointer: UnsafeMutableRawPointer?
        let f16Size = MemoryLayout<Float16>.stride

        _ = aPointer.cudaMemoryAllocate(m * k * f16Size)
        _ = bPointer.cudaMemoryAllocate(k * n * f16Size)
        _ = cPointer.cudaMemoryAllocate(m * n * f16Size)
        defer {
            _ = aPointer.cudaAndHostDeallocate()
            _ = bPointer.cudaAndHostDeallocate()
            _ = cPointer.cudaAndHostDeallocate()
        }
        _ = aPointer.cudaMemoryCopy(fromRawPointer: &A, numberOfBytes: A.count * f16Size, copyKind: .cudaMemcpyHostToDevice)
        _ = bPointer.cudaMemoryCopy(fromRawPointer: &B, numberOfBytes: B.count * f16Size, copyKind: .cudaMemcpyHostToDevice)
        let desc = CUBLASLtMatMulDescriptor(computeType: .cublas_compute_16f, dataType: .CUDA_R_16F)
        let aDesc = CUBLASLtMatrixLayout(fromRowMajor: .CUDA_R_16F, rows: m, columns: k, ld: k)
        let bDesc = CUBLASLtMatrixLayout(fromRowMajor: .CUDA_R_16F, rows: k, columns: n, ld: n)
        let cDesc = CUBLASLtMatrixLayout(fromRowMajor: .CUDA_R_16F, rows: m, columns: n, ld: n)
        var params = CUBLASLtaramsMixed(
            A: aPointer!.assumingMemoryBound(to: Float16.self), B: bPointer!.assumingMemoryBound(to: Float16.self),
            C: cPointer!.assumingMemoryBound(to: Float16.self), aDesc: aDesc, bDesc: bDesc, cDesc: cDesc, alpha: Float16(1.0), beta: Float16(0.0)
         )
        let status = handle.gemm(desc: desc, params: &params)
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(
                fromMutableRawPointer: cPointer, numberOfBytes: m * n * f16Size, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let cExpected = matrixMultiply(m, n, k, A, B, isRowMajor: true)
        XCTAssert(cExpected.map { Float16($0) } ~= C)
    }

    func testMatmulF32() throws {
        let handle = CUBLASLtHandle()
        _ = CUDADevice(index: 0).setDevice()
        let m = 2
        let n = 2
        let k = 4

        var A: [Float32] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]

        var B: [Float32] = [
            8.0, 7.0,
            6.0, 5.0,
            4.0, 3.0,
            2.0, 1.0,
        ]

        var C = [Float32](repeating: 0.0, count: m * n)

        var aPointer: UnsafeMutableRawPointer?
        var bPointer: UnsafeMutableRawPointer?
        var cPointer: UnsafeMutableRawPointer?
        let typeSize: Int = MemoryLayout<Float32>.stride

        _ = aPointer.cudaMemoryAllocate(m * k * typeSize)
        _ = bPointer.cudaMemoryAllocate(k * n * typeSize)
        _ = cPointer.cudaMemoryAllocate(m * n * typeSize)
        defer {
            _ = aPointer.cudaAndHostDeallocate()
            _ = bPointer.cudaAndHostDeallocate()
            _ = cPointer.cudaAndHostDeallocate()
        }
        let computeType: cublasComputeType = .cublas_compute_32f
        let inputType: swiftCUDADataType = .CUDA_R_32F
        _ = aPointer.cudaMemoryCopy(fromRawPointer: &A, numberOfBytes: A.count * typeSize, copyKind: .cudaMemcpyHostToDevice)
        _ = bPointer.cudaMemoryCopy(fromRawPointer: &B, numberOfBytes: B.count * typeSize, copyKind: .cudaMemcpyHostToDevice)
        let desc = CUBLASLtMatMulDescriptor(computeType: computeType, dataType: inputType)
        let aDesc = CUBLASLtMatrixLayout(fromRowMajor: inputType, rows: m, columns: k, ld: k)
        let bDesc = CUBLASLtMatrixLayout(fromRowMajor: inputType, rows: k, columns: n, ld: n)
        let cDesc = CUBLASLtMatrixLayout(fromRowMajor: inputType, rows: m, columns: n, ld: n)
        var params = CUBLASLtaramsMixed(
            A: aPointer!.assumingMemoryBound(to: Float32.self), B: bPointer!.assumingMemoryBound(to: Float32.self),
            C: cPointer!.assumingMemoryBound(to: Float32.self), aDesc: aDesc, bDesc: bDesc, cDesc: cDesc, alpha: Float32(1.0), beta: Float32(0.0)
         )
        let status = handle.gemm(desc: desc, params: &params)
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(
                fromMutableRawPointer: cPointer, numberOfBytes: m * n * typeSize, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let cExpected = matrixMultiply(m, n, k, A, B, isRowMajor: true)
        XCTAssert(cExpected ~= C)
    }

    func testMatmulF16F32() throws {
        let handle = CUBLASLtHandle()
        _ = CUDADevice(index: 0).setDevice()
        let m = 2
        let n = 2
        let k = 4

        var A: [Float16] = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]

        var B: [Float16] = [
            8.0, 7.0,
            6.0, 5.0,
            4.0, 3.0,
            2.0, 1.0,
        ]

        var C = [Float32](repeating: 0.0, count: m * n)

        var aPointer: UnsafeMutableRawPointer?
        var bPointer: UnsafeMutableRawPointer?
        var cPointer: UnsafeMutableRawPointer?
        let typeSize: Int = MemoryLayout<Float32>.stride
        let outTypeSize: Int = MemoryLayout<Float32>.stride

        _ = aPointer.cudaMemoryAllocate(m * k * typeSize)
        _ = bPointer.cudaMemoryAllocate(k * n * typeSize)
        _ = cPointer.cudaMemoryAllocate(m * n * outTypeSize)
        defer {
            _ = aPointer.cudaAndHostDeallocate()
            _ = bPointer.cudaAndHostDeallocate()
            _ = cPointer.cudaAndHostDeallocate()
        }
        _ = aPointer.cudaMemoryCopy(fromRawPointer: &A, numberOfBytes: A.count * typeSize, copyKind: .cudaMemcpyHostToDevice)
        _ = bPointer.cudaMemoryCopy(fromRawPointer: &B, numberOfBytes: B.count * typeSize, copyKind: .cudaMemcpyHostToDevice)
        let desc = CUBLASLtMatMulDescriptor(computeType: .cublas_compute_32f, dataType: .CUDA_R_32F)
        let aDesc = CUBLASLtMatrixLayout(fromRowMajor: .CUDA_R_16F, rows: m, columns: k, ld: k)
        let bDesc = CUBLASLtMatrixLayout(fromRowMajor: .CUDA_R_16F, rows: k, columns: n, ld: n)
        let cDesc = CUBLASLtMatrixLayout(fromRowMajor: .CUDA_R_32F, rows: m, columns: n, ld: n)
        var params = CUBLASLtaramsMixed(
            A: aPointer!.assumingMemoryBound(to: Float16.self), B: bPointer!.assumingMemoryBound(to: Float16.self),
            C: cPointer!.assumingMemoryBound(to: Float32.self), aDesc: aDesc, bDesc: bDesc, cDesc: cDesc, alpha: Float32(1.0), beta: Float32(0.0)
         )
        let status = handle.gemm(desc: desc, params: &params)
        XCTAssert(status.isSuccessful)
        C.withUnsafeMutableBytes { rawBufferPointer in
            var pointerAddress = rawBufferPointer.baseAddress
            let outStatus = pointerAddress.cudaMemoryCopy(
                fromMutableRawPointer: cPointer, numberOfBytes: m * n * outTypeSize, copyKind: .cudaMemcpyDeviceToHost)
            XCTAssert(outStatus.isSuccessful)
        }
        cudaDeviceSynchronize()
        let cExpected = matrixMultiply(m, n, k, A, B, isRowMajor: true)
        XCTAssert(cExpected.map{Float32($0)} ~= C)
    }
}

func getIndex(row: Int, col: Int, numRows: Int, numCols: Int, isRowMajor: Bool) -> Int {
    return isRowMajor ? row * numCols + col : col * numRows + row
}

func matrixMultiply<T: CUBLASDataType & Numeric>(
    _ m: Int,
    _ n: Int,
    _ k: Int,
    _ A: [T],
    _ B: [T],
    isRowMajor: Bool
) -> [T] {
    var C: [T] = [T](repeating: 0, count: m * n)
    for i in 0..<m {
        for j in 0..<n {
            var sum: T = 0
            for p in 0..<k {
                let aIndex = getIndex(row: i, col: p, numRows: m, numCols: k, isRowMajor: isRowMajor)
                let bIndex = getIndex(row: p, col: j, numRows: k, numCols: n, isRowMajor: isRowMajor)
                sum += A[aIndex] * B[bIndex]
            }
            let cIndex = getIndex(row: i, col: j, numRows: m, numCols: n, isRowMajor: isRowMajor)
            C[cIndex] = sum
        }
    }
    return C
}
