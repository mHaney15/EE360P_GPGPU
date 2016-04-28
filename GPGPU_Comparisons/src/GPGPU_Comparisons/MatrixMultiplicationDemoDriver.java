package GPGPU_Comparisons;
//DEMO PURPOSE ONLY!
import org.lwjgl.opencl.Util;
import org.lwjgl.opencl.CLMem;
import org.lwjgl.opencl.CLCommandQueue;
import org.lwjgl.BufferUtils;
import org.lwjgl.LWJGLException;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CLProgram;
import org.lwjgl.opencl.CLKernel;
import java.io.IOException;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLContext;
import org.lwjgl.opencl.CLDevice;
import org.lwjgl.opencl.CLPlatform;
import static org.lwjgl.opencl.CL10.*;

public class MatrixMultiplicationDemoDriver {
	
	static double[][] run(double[][] A, double[][] B){
		
		double[][]GPU_Result = null;
		try {
			long startSetup = System.nanoTime();
			int SIZE = A.length;
			
			//Convert Java Objects to Buffers, and allocate a buffer for receiving the answer.
			DoubleBuffer a = toDoubleBuffer(A); 
			DoubleBuffer b = toDoubleBuffer(B);
			DoubleBuffer answer = BufferUtils.createDoubleBuffer(a.capacity());
			
			//Read in the OpenCL kernel as a string
			String kernelString = readFile("src/Mat_Mult.cl");
			
			// Initialize OpenCL and create a context and command queue
			IntBuffer errorBuff = BufferUtils.createIntBuffer(1);
			CL.create();
			CLPlatform platform = CLPlatform.getPlatforms().get(1); //<- for me, platform 1 is NVIDIA 
			List<CLDevice> devices = platform.getDevices(CL10.CL_DEVICE_TYPE_GPU); 
			CLContext context = CLContext.create(platform, devices, errorBuff);
			CLCommandQueue queue = CL10.clCreateCommandQueue(context, devices.get(0), //<-Device 1 is my 940m
								   CL10.CL_QUEUE_PROFILING_ENABLE, errorBuff);
			Util.checkCLError(errorBuff.get(0));
			
			// Allocate memory for our two input buffers and our result buffer in actual memory
			CLMem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a, null);
	        clEnqueueWriteBuffer(queue, aMem, 1, 0, a, null, null);
	        CLMem bMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b, null);
	        clEnqueueWriteBuffer(queue, bMem, 1, 0, b, null, null);
	        CLMem answerMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, answer, null);
	        clFinish(queue);	        
	        CLProgram program = clCreateProgramWithSource(context, kernelString, null);
	        Util.checkCLError(clBuildProgram(program, devices.get(0), "", null));
	        CLKernel kernel = clCreateKernel(program, "Mat_Mult", null); //<- the name of our kernel function.
	 
	        // Execution our kernel
	        //Create a 2D workspace (You can also use 1 or 3D!)
	        PointerBuffer kernel2DGlobalWorkSize = BufferUtils.createPointerBuffer(2);
	        kernel2DGlobalWorkSize.put(0, SIZE); //We want a SIZExSIZE workspace
	        kernel2DGlobalWorkSize.put(1, SIZE);
	        kernel.setArg(0, aMem);				//Add pointers and primitive size as kernel args
	        kernel.setArg(1, bMem);
	        kernel.setArg(2, answerMem);
	        kernel.setArg(3, SIZE);
	        long startGPUExecution = System.nanoTime();
	        clEnqueueNDRangeKernel(queue, kernel, 2, null, kernel2DGlobalWorkSize, null, null, null);
	        long endGPUExecution = System.nanoTime();
	        
	        // Read the results memory back into our result buffer
	        clEnqueueReadBuffer(queue, answerMem, 1, 0, answer, null, null);
	        clFinish(queue);
	        GPU_Result = RMVG.BufferToMatrix(answer, SIZE);
	        
	     // Clean up OpenCL resources
	        clReleaseKernel(kernel);
	        clReleaseProgram(program);
	        clReleaseMemObject(aMem);
	        clReleaseMemObject(bMem);
	        clReleaseMemObject(answerMem);
	        clReleaseCommandQueue(queue);
	        clReleaseContext(context);
	        CL.destroy();     
	        long EndConversion = System.nanoTime();
	        if(SIZE > 1000){
	        	DemoDriver.GPU_RT = EndConversion - startSetup;
	        	DemoDriver.GPU_ET = endGPUExecution - startGPUExecution;
	        	
	        }
	        
	        return GPU_Result;
	             
	        
		} catch (LWJGLException e) {
			System.err.println("Looks like your system does not support OpenCL. :(");
			e.printStackTrace();
		} catch (IOException e){
			System.err.println("Oh no, there was an issue reading in your kernel!");
			e.printStackTrace();
		}
		finally{return GPU_Result;}
	}
	
	/** Utility method to convert double array to double buffer
     * @param doubles - the double array to convert
     * @return a double buffer containing the input double array
     */
    static DoubleBuffer toDoubleBuffer(double[] doubles) {
        DoubleBuffer buf = BufferUtils.createDoubleBuffer(doubles.length).put(doubles);
        buf.rewind();
        return buf;
    }
    
    /** Utility method to convert double matrices to double buffer
     * @param doubles - the double matrix to convert
     * @return a double buffer containing the input double matrix
     */
    static DoubleBuffer toDoubleBuffer(double[][] doubles) {
        DoubleBuffer buf = BufferUtils.createDoubleBuffer(doubles.length*doubles[0].length);
        for(int i = 0; i < doubles.length; i++){
        	buf.put(doubles[i]);
        }
        buf.rewind();
        return buf;
    }
 
 
    /** Utility method to print a double buffer that is a Vector
     * @param buffer - the double buffer to print to System.out
     */
    static void printVector(DoubleBuffer buffer) {
        for (int i = 0; i < buffer.capacity(); i++) {
            System.out.print(buffer.get(i)+" ");
        }
        System.out.println("");
    }
    
    /** Utility method to print a double buffer that is a Matrix
     * @param buffer - the double buffer to print to System.out
     */
    static void printMatrix(DoubleBuffer buffer, int w) {
        for (int i = 0; i < buffer.capacity(); i++) {
            if((i%w == 0)&&(i > 0)){
            	System.out.print("\n");
            }
        	System.out.print(buffer.get(i)+" ");
        }
        System.out.println("");
    }
    
    //Converts a file to a string using standard UTF-8 Encoding.
    static String readFile(String path) throws IOException 
    {
    	byte[] encoded = Files.readAllBytes(Paths.get(path));
    	return new String(encoded, "UTF-8");
   	}   
    
    public static void displayInfo() {

        for (int platformIndex = 0; platformIndex < CLPlatform.getPlatforms().size(); platformIndex++) {
            CLPlatform platform = CLPlatform.getPlatforms().get(platformIndex);
            System.out.println("Platform #" + platformIndex + ":" + platform.getInfoString(CL_PLATFORM_NAME));
            List<CLDevice> devices = platform.getDevices(CL_DEVICE_TYPE_ALL);
            for (int deviceIndex = 0; deviceIndex < devices.size(); deviceIndex++) {
                CLDevice device = devices.get(deviceIndex);
                System.out.printf(Locale.ENGLISH, "Device #%d(%s):%s\n",
                        deviceIndex,
                        UtilCL.getDeviceType(device.getInfoInt(CL_DEVICE_TYPE)),
                        device.getInfoString(CL_DEVICE_NAME));
                System.out.printf(Locale.ENGLISH, "\tCompute Units: %d @ %d mghtz\n",
                        device.getInfoInt(CL_DEVICE_MAX_COMPUTE_UNITS), device.getInfoInt(CL_DEVICE_MAX_CLOCK_FREQUENCY));
                System.out.printf(Locale.ENGLISH, "\tLocal memory: %s\n",
                        UtilCL.formatMemory(device.getInfoLong(CL_DEVICE_LOCAL_MEM_SIZE)));
                System.out.printf(Locale.ENGLISH, "\tGlobal memory: %s\n",
                        UtilCL.formatMemory(device.getInfoLong(CL_DEVICE_GLOBAL_MEM_SIZE)));
                System.out.println();
            }
        }
    }
    
    public static void printMatrix(double[][] m){
        try{
            int rows = m.length;
            int columns = m[0].length;
            String str = "|\t";

            for(int i=0;i<rows;i++){
                for(int j=0;j<columns;j++){
                    str += m[i][j] + "\t";
                }

                System.out.println(str + "|");
                str = "|\t";
            }

        }catch(Exception e){System.out.println("Matrix is empty!!");}
    }
}
