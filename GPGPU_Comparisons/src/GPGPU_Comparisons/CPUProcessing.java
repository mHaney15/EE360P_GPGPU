package GPGPU_Comparisons;

import java.util.Arrays;

public class CPUProcessing {
	/* Assumes a and b are vectors/arrays with the same length. */
	public static double[] vectorAddition(double[] a, double[] b) {
		double[] retval = new double[a.length];
		for(int i = 0; i < a.length; i += 1) {
			retval[i] = a[i] + b[i];
		}
		return retval;
	}
	
	/* Assumes a is a square matrix that is invertible (has non-zero determinant). */
	public static double[][] matrixInversion(double[][] a) {
		// Code for this function and the functions it calls was adapted from
		// http://mrbool.com/how-to-use-java-for-performing-matrix-operations/26800
		
		return (divideBy(transpose(cofactor(a)), determinant(a)));
	}
	
	private static double[][] divideBy(double[][] a, double x) {
		for (int i = 0; i < a.length; i += 1) {
			for (int j = 0; j < a.length; j += 1) {
				a[i][j] /= x;
			}
		}
		return a;
	}
	
	public static double[][] transpose(double[][] a) {
	    double[][] retval = new double[a.length][a[0].length];
	    
	    for (int i = 0 ;i < a.length; i += 1) {
	        for (int j = 0; j < a.length; j += 1) {
	        	retval[j][i] = a[i][j];
	        }
	    }
	    return retval;
	}
	
	private static double changeSign(double a) {
		return (a%2 == 0) ? 1 : -1;
	}
	
	private static double determinant(double[][] a) {
		
		if (a.length == 2) {
	        return (a[0][0] * a[1][1]) - (a[0][1] * a[1][0]);
	    }
		
	    double sum = 0;
	    for (int i = 0; i < a.length; i += 1) {
	    	double sign = changeSign(i);
	    	double multiplier = a[0][i];
	    	double babyDeterminant = determinant(createSubMatrix(a, 0, i));
	        //sum += changeSign(i) * a[0][i] * determinant(createSubMatrix(a, 0, i));
	    	sum += (sign*multiplier*babyDeterminant);
	    }
	    return sum;
	}
	
	private static double[][] createSubMatrix(double[][] a, double excluding_row, double excluding_col) {
		
	    double[][] retval = new double[a.length - 1][a.length - 1];
	    int r = -1;
	    
	    for (int i = 0; i < a.length; i += 1) {
	        if (i == excluding_row)
	            continue;
            r++;
            int c = -1;
	        for (int j = 0; j < a.length; j += 1) {
	            if (j==excluding_col)
	                continue;
	            retval[r][++c] = a[i][j];
	        }
	    }
	    return retval;
	}
	
	private static double sign(double i) {
		return (i < 0) ? -1 : 1;
	}
	
	private static double[][] cofactor(double[][] a) {
		double[][] retval = new double[a.length][a.length];
	    for (int i = 0; i < a.length; i += 1) {
	        for (int j = 0; j < a.length; j += 1) {
	            retval[i][j] = sign(i) * changeSign(j) * determinant(createSubMatrix(a, i, j));
	        }
	    }
	    
	    return retval;
	}
	
	public static double[][] matrixMultiplication(double[][] a, double[][] b) {
		// Code adopted from
		// http://stackoverflow.com/questions/17623876/matrix-multiplication-using-arrays
		double[][] retval = new double[a.length][a.length];
		
		for (int i = 0; i < a.length; i += 1) {
			for (int j = 0; j < a.length; j += 1) {
				for (int k = 0; k < a.length; k += 1) {
					retval[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		
		return retval;
	}
	
	public static String encrypt(String data, int key) {
		char[] retval = data.toCharArray();
		
		for(int i = 0; i < retval.length; i += 1) {
			retval[i] ^= key;
		}
		
		return new String(retval);
	}
	
	public static String decrypt(String data, int key) {
		return encrypt(data, key);
	}
	
	public static void main(String[] args) {
		System.out.println("Testing CPU processing methods.");
		
		double[] inputA = {1,2,3,4,5};
		double[] inputB = {1,2,3,4,5};
		double[] output = vectorAddition(inputA, inputB);
		
		System.out.println(Arrays.toString(inputA) + " * 2 = " + Arrays.toString(output));
		
		double[][] matrixA = {{1,3,5,9},
						   	  {1,3,1,7},
						   	  {4,3,9,7},
						   	  {5,2,0,9}};
		
		System.out.println(determinant(matrixA));
		double[][] output2 = matrixInversion(matrixA);
		System.out.println(Arrays.toString(output2[0]));
		System.out.println(Arrays.toString(output2[1]));
		System.out.println(Arrays.toString(output2[2]));
		System.out.println(Arrays.toString(output2[3]));
		
		double[][] output3 = matrixMultiplication(matrixA, matrixA);
		System.out.println(Arrays.toString(output3[0]));
		System.out.println(Arrays.toString(output3[1]));
		System.out.println(Arrays.toString(output3[2]));
		System.out.println(Arrays.toString(output3[3]));
		
		String test = "The cow jumped over the moon, and you liked it.";
		String encodedTest = encrypt(test, 0x02349802);
		System.out.println("Encoding: " + test);
		System.out.println("Encoded: " + encodedTest);
		System.out.println("Decoded: " + decrypt(encodedTest, 0x02349802));
		
		System.out.println("CPU testing complete!");
	}
}
