package GPGPU_Comparisons;

import java.util.HashMap;
import java.util.Map;

public class Driver {

	private final static int NUM_ITERATIONS = 100;
	private final static int MAX_VECTOR_SIZE = 4096;
	
	public static void main(String[] args) {
		testVectorAdd();
	}
	
	private static void testVectorAdd() {
		HashMap<Integer, TimeResults> vector_add_results = new HashMap<Integer, TimeResults>();
		
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
		for(Map.Entry entry : vector_add_results.entrySet()) {
			TimeResults tr = (TimeResults) entry.getValue();
			System.out.println("<size = " + entry.getKey() +">");
			System.out.println("\tGPU OpenCL Overhead: " + tr.GPU_OpenCL_Overhead);
			System.out.println("\tGPU Data Transfer: " + tr.GPU_DataTransfer);
			System.out.println("\tGPU Exec: " + tr.GPU_Exec);
			System.out.println("\tCPU Exec: " + tr.CPU_Exec);
		}
		System.out.println("----VECTOR ADD PERFORMANCE----");
	}
}
