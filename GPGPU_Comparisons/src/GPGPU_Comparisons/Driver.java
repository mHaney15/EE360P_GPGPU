package GPGPU_Comparisons;

import java.io.File;
import java.io.PrintWriter;
import java.io.WriteAbortedException;
import java.util.HashMap;
import java.util.Map;

public class Driver {

	private final static int NUM_ITERATIONS = 100;
	private final static int MAX_VECTOR_SIZE = 524288;
	private final static int MAX_MATRIX_SIZE = 1024;
	
	public static void main(String[] args) {
		try{
			File results = new File("Results.csv");
			if(results.exists()){results.delete();}
			else{results.createNewFile();}
			results.setWritable(true);
			PrintWriter writer = new PrintWriter("Results.csv", "UTF-8");
			writer.write("Operation, Size, OpenCL Overhead, GPU Data Trans, GPU Ex, CPU Ex,\n");
			
			HashMap<Integer, TimeResults> vector_add_results = new HashMap<Integer, TimeResults>();
			HashMap<Integer, TimeResults> matrix_mult_results = new HashMap<Integer, TimeResults>(); 
			HashMap<Integer, TimeResults> matrix_transpose_results = new HashMap<Integer, TimeResults>();
			HashMap<Integer, TimeResults> string_encryption_results = new HashMap<Integer, TimeResults>();
			
			for (int size = 32; size <= MAX_VECTOR_SIZE; size *= 2) {
				TimeResults avg = new TimeResults();
				System.out.println("Executing Vector Addition with size "+size+"...");
				for (int i = 0; i < NUM_ITERATIONS; i += 1) {
					TimeResults result = GPGPGU_VECTADD_Driver.run(size);
					avg.add(result);
					writer.write("Vector Addition, "+size+", "+result.GPU_OpenCL_Overhead+", "+result.GPU_DataTransfer+" ,"+result.GPU_Exec+", "+result.CPU_Exec+",\n");					
					writer.flush();
				}
				
				avg.divideAll(NUM_ITERATIONS);
				vector_add_results.put(size, avg);
			}
			
			System.out.println("----VECTOR ADD PERFORMANCE----");
			printResults(vector_add_results);
			System.out.println("------------------------------\n");
			
			for (int size = 32; size <= MAX_MATRIX_SIZE; size *= 2) {
				TimeResults avg = new TimeResults();
				System.out.println("Executing Matrix Multiplication with size "+size+"...");
				for (int i = 0; i < NUM_ITERATIONS; i += 1) {
					TimeResults result = GPGPGU_MATMULT_Driver.run(size);
					avg.add(result);
					writer.write("Matrix Multiplication, "+size+", "+result.GPU_OpenCL_Overhead+", "+result.GPU_DataTransfer+" ,"+result.GPU_Exec+", "+result.CPU_Exec+",\n");
					writer.flush();
				}
				
				avg.divideAll(NUM_ITERATIONS);
				matrix_mult_results.put(size, avg);
			}
			
			System.out.println("----MATRIX MULTIPLICATION PERFORMANCE----");
			printResults(matrix_mult_results);
			System.out.println("------------------------------\n");
			
			for (int size = 32; size <= MAX_MATRIX_SIZE; size *= 2) {
				TimeResults avg = new TimeResults();
				System.out.println("Executing Matrix Transpose with size "+size+"...");
				for (int i = 0; i < NUM_ITERATIONS; i += 1) {
					TimeResults result = GPGPGU_MATTRANS_Driver.run(size);
					avg.add(result);
					writer.write("Matrix Transpose, "+size+", "+result.GPU_OpenCL_Overhead+", "+result.GPU_DataTransfer+" ,"+result.GPU_Exec+", "+result.CPU_Exec+",\n");
					writer.flush();
				}
				
				avg.divideAll(NUM_ITERATIONS);
				matrix_transpose_results.put(size, avg);
			}
			
			System.out.println("----MATRIX TRANSPOSE PERFORMANCE----");
			printResults(matrix_transpose_results);
			System.out.println("------------------------------\n");
			
			for (int size = 32; size <= MAX_MATRIX_SIZE; size *= 2) {
				TimeResults avg = new TimeResults();
				System.out.println("Executing Encryption with size "+size+"...");
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
		
			writer.flush();
			writer.close();
		} catch(Exception e){
			System.err.println("Something bad happened somewhere...");
			e.printStackTrace();
		}
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
