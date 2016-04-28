package GPGPU_Comparisons;

import java.util.Scanner;

public class DemoDriver {
	
	public static long GPU_RT;	//GPU Full Runtime
	public static long GPU_ET;	//GPU Execution Time
	public static long CPU_ET;	//CPU Execution Time
	
	public static void main(String[] args){
		
		//First Example
		double[][] A = RMVG.getRandomMatrix(4);
		double[][] B = RMVG.getRandomMatrix(4);
		double[][] C = MatrixMultiplicationDemoDriver.run(A, B);
		double[][] D = CPUProcessing.matrixMultiplication(A, B);
		
		System.out.println("Matrix A:");
		MatrixMultiplicationDemoDriver.printMatrix(A);
		System.out.println("\nMatrix B:");
		MatrixMultiplicationDemoDriver.printMatrix(B);
		System.out.println("\nMatrix AxB GPU Result:");
		MatrixMultiplicationDemoDriver.printMatrix(C);
		System.out.println("\nMatrix AxB CPU Result:");
		MatrixMultiplicationDemoDriver.printMatrix(D);
		
		if(RMVG.ResultsEqual(C, D)){System.out.println("\nThey Matched! (at least within 0.00000001%)\n");}
		
		Scanner s = new Scanner(System.in);
		System.out.print("Continue with 1024x1024 example? (Yes/No) ");
		String input = s.nextLine();
		if(input.equals("No")){System.exit(0);}
		
		System.out.println("\nCreating two 1024x1024 matrices...");
		A = RMVG.getRandomMatrix(1024);
		B = RMVG.getRandomMatrix(1024);
		
		System.out.println("Starting GPU Implementation...");
		C = MatrixMultiplicationDemoDriver.run(A, B);
		System.out.println("\tComplete!");
		
		System.out.println("Starting CPU Implementation...");
		long startCPU = System.nanoTime();
		D = CPUProcessing.matrixMultiplication(A, B);
		System.out.println("\tComplete!");
		long endCPU = System.nanoTime();
		CPU_ET = endCPU - startCPU;
		
		System.out.println("\nGPU Runtime with LWJGL Overhead: "+GPU_RT+" ns.");
    	System.out.println("GPU Pure Execution Time: "+GPU_ET+" ns.");
    	System.out.println("CPU Execution Time: "+CPU_ET+" ns.");
    	System.out.println("Checking for Equality...");
    	if(RMVG.ResultsEqual(C, D)){System.out.println("They Matched! (at least within 0.00000001%)");}

    	s.close();
	}
}
