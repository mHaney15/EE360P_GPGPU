package GPGPU_Comparisons;

import java.util.HashMap;
import java.util.Map;

public class Driver {

	private final static int NUM_ITERATIONS = 100;
	private final static int MAX_VECTOR_SIZE = 4096;
	private final static int MAX_MATRIX_SIZE = 1024;
	
	public static void main(String[] args) {
		HashMap<Integer, TimeResults> vector_add_results = new HashMap<Integer, TimeResults>();
		HashMap<Integer, TimeResults> matrix_mult_results = new HashMap<Integer, TimeResults>(); 
		HashMap<Integer, TimeResults> matrix_transpose_results = new HashMap<Integer, TimeResults>();
		HashMap<Integer, TimeResults> string_encryption_results = new HashMap<Integer, TimeResults>();
		
		for (int size = 32; size <= MAX_VECTOR_SIZE; size *= 2) {
			TimeResults avg = new TimeResults();
			for (int i = 0; i < NUM_ITERATIONS; i += 1) {
				TimeResults result = GPGPGU_VECTADD_Driver.run(size);
				avg.add(result);
			}
			
			avg.divideAll(NUM_ITERATIONS);
			vector_add_results.put(size, avg);
		}
		
		System.out.println("----VECTOR ADD PERFORMANCE----");
		printResults(vector_add_results);
		System.out.println("------------------------------\n");
		
		for (int size = 32; size <= MAX_MATRIX_SIZE; size *= 2) {
			TimeResults avg = new TimeResults();
			for (int i = 0; i < NUM_ITERATIONS; i += 1) {
				TimeResults result = GPGPGU_MATMULT_Driver.run(size);
				avg.add(result);
			}
			
			avg.divideAll(NUM_ITERATIONS);
			matrix_mult_results.put(size, avg);
		}
		
		System.out.println("----MATRIX MULTIPLICATION PERFORMANCE----");
		printResults(matrix_mult_results);
		System.out.println("------------------------------\n");
		
		for (int size = 32; size <= MAX_MATRIX_SIZE; size *= 2) {
			TimeResults avg = new TimeResults();
			for (int i = 0; i < NUM_ITERATIONS; i += 1) {
				TimeResults result = GPGPGU_MATTRANS_Driver.run(size);
				avg.add(result);
			}
			
			avg.divideAll(NUM_ITERATIONS);
			matrix_transpose_results.put(size, avg);
		}
		
		System.out.println("----MATRIX TRANSPOSE PERFORMANCE----");
		printResults(matrix_transpose_results);
		System.out.println("------------------------------\n");
		
		for (int size = 32; size <= MAX_MATRIX_SIZE; size *= 2) {
			TimeResults avg = new TimeResults();
			for (int i = 0; i < NUM_ITERATIONS; i += 1) {
				TimeResults result = GPGPGU_XORENCRYPT_Driver.run();
				avg.add(result);
			}
			
			avg.divideAll(NUM_ITERATIONS);
			string_encryption_results.put(size, avg);
		}
		
		System.out.println("----STRING ENCRYPTION PERFORMANCE----");
		printResults(string_encryption_results);
		System.out.println("------------------------------\n");
	}
	
	private static void printResults(HashMap<Integer, TimeResults> map) {
		for(Map.Entry entry : map.entrySet()) {
			TimeResults tr = (TimeResults) entry.getValue();
			System.out.println("<size = " + entry.getKey() +">");
			System.out.println("\tGPU OpenCL Overhead: " + tr.GPU_OpenCL_Overhead);
			System.out.println("\tGPU Data Transfer: " + tr.GPU_DataTransfer);
			System.out.println("\tGPU Exec: " + tr.GPU_Exec);
			System.out.println("\tCPU Exec: " + tr.CPU_Exec);
		}
	}
}
