package GPGPU_Comparisons;

public class CPUProcessing {
	/* Assumes a and b are vectors/arrays with the same length. */
	public static int[] vectorAddition(int[] a, int[] b) {
		int[] retval = new int[a.length];
		for(int i = 0; i < a.length; i += 1) {
			retval[i] = a[i] + b[i];
		}
		return retval;
	}
	
	/* Assumes a is a square matrix that is invertible (has non-zero determinant). */
	public static int[][] matrixInversion(int[][] a) {
		// Code for this function and the functions it calls was adapted from
		// http://mrbool.com/how-to-use-java-for-performing-matrix-operations/26800
		
		return (divideBy(transpose(cofactor(a)), determinant(a)));
	}
	
	private static int[][] divideBy(int[][] a, int x) {
		for (int i = 0; i < a.length; i += 1) {
			for (int j = 0; j < a.length; j += 1) {
				a[i][j] /= x;
			}
		}
		return a;
	}
	
	private static int[][] transpose(int[][] a) {
	    int[][] retval = new int[a.length][a[0].length];
	    
	    for (int i = 0 ;i < a.length; i += 1) {
	        for (int j = 0; j < a.length; j += 1) {
	        	retval[j][i] = a[i][j];
	        }
	    }
	    return retval;
	}
	
	private static int changeSign(int a) {
		return (a%2 == 0) ? 1 : -1;
	}
	
	private static int determinant(int[][] a) {
	    int sum = 0;
	    for (int i = 0; i < a.length; i += 1) {
	    	sum += 
	        sum += changeSign(i) * a[0][i] * determinant(createSubMatrix(a, 0, i));
	    }
	    return sum;
	}
	
	private static int[][] createSubMatrix(int[][] a, int excluding_row, int excluding_col) {
		
	    int[][] retval = new int[a.length - 1][a.length - 1];
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
	
	private static int sign(int i) {
		return (i < 0) ? -1 : 1;
	}
	
	private static int[][] cofactor(int[][] a) {
		int[][] retval = new int[a.length][a.length];
	    for (int i = 0; i < a.length; i += 1) {
	        for (int j = 0; j < a.length; j += 1) {
	            retval[i][j] = sign(i) * changeSign(j) * determinant(createSubMatrix(a, i, j));
	        }
	    }
	    
	    return retval;
	}
	
	/* W */
	public static int[][] matrixMultiplication(int[][] a, int[][] b) {
		// Code adopted from
		// http://stackoverflow.com/questions/17623876/matrix-multiplication-using-arrays
		int[][] retval = new int[a.length][a.length];
		
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
		
	}
}
