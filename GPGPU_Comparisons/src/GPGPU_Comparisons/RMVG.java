package GPGPU_Comparisons;

import java.util.Random;
import java.nio.DoubleBuffer;

public class RMVG {
	static Random RNG = new Random();
	
	public static double[] getRandomVector(int size){
		double[] randomArray = new double[size];
		for(int i = 0; i < size; i++){
			randomArray[i] = RNG.nextDouble()*(Double.MAX_VALUE/2);
		}
		return randomArray;
	}
	
	public static double[][] getRandomMatrix(int size){
		double[][] randomMatrix = new double[size][size];
		for(int i = 0; i < size; i++){
			for(int j = 0; j < size; j++){
				randomMatrix[i][j] = RNG.nextDouble()*(Double.MAX_VALUE/size);
			}
		}
		return randomMatrix;
	}
	
	public static double[] BufferToArray(DoubleBuffer dbuffer){
		double[] D_Array = new double[dbuffer.capacity()];
		for(int i = 0; i < D_Array.length; i++){
			D_Array[i] = dbuffer.get(i);
		}
		return D_Array;
	}
	
	public static double[][] BufferToMatrix(DoubleBuffer dbuffer, int Width){
		double[][] D_Matrix = new double[dbuffer.capacity()/Width][Width];
		for(int i = 0; i < dbuffer.capacity(); i++){
			D_Matrix[i/Width][i%Width] = dbuffer.get(i);
		}
		return D_Matrix;
	}
	
	public static boolean CompareResults(double[] a1, double[] a2){
		if(a1.length != a2.length){return false;}
		for(int i = 0; i < a1.length; i++){
			if(a1[i] != a2[i]){
				System.out.println("Element number "+i+" in the arrays differed by "+Math.abs(a1[i]-a2[i])+"!");
				return false;
			}
		}
		return true;
	}

}
