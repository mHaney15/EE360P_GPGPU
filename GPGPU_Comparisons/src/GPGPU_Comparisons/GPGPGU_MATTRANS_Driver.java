package GPGPU_Comparisons;

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

public class GPGPGU_MATTRANS_Driver {
	public static CLContext context;
	public static CLPlatform platform;
	public static List<CLDevice> devices;
	public static CLCommandQueue queue;
	static DoubleBuffer a;
	static DoubleBuffer answer;
	static String kernelString; 
	
	public static TimeResults run(int SIZE){
		TimeResults retval = new TimeResults();
		try {
			
//			System.out.println("Creating random matrix A...");
			double[][] A = RMVG.getRandomMatrix(SIZE);
			
			long Begin_Init = System.nanoTime();
			//Read in Kernel file.
			kernelString = readFile("src/Mat_Trans.cl");
			initializeCL();
			long End_Init = System.nanoTime();
			
			//Need to create matrices/vectors here!
			long Begin_DataTransfer = System.nanoTime();
			a = toDoubleBuffer(A);
			answer = BufferUtils.createDoubleBuffer(a.capacity());
			
			// Allocate memory for our two input buffers and our result buffer
			CLMem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a, null);
	        clEnqueueWriteBuffer(queue, aMem, 1, 0, a, null, null);
	        CLMem answerMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, answer, null);
	        clFinish(queue);
	        CLProgram program = clCreateProgramWithSource(context, kernelString, null);
//	        displayInfo();
//	        clBuildProgram(program, devices.get(0), "", null);
//	        System.out.println(program.getBuildInfoString(
//	        devices.get(0), CL_PROGRAM_BUILD_LOG));
	        Util.checkCLError(clBuildProgram(program, devices.get(0), "", null));
	        // sum has to match a kernel method name in the OpenCL source
	        CLKernel kernel = clCreateKernel(program, "Mat_Trans", null);
	 
	        // Execution our kernel
	        PointerBuffer kernel2DGlobalWorkSize = BufferUtils.createPointerBuffer(2);
	        kernel2DGlobalWorkSize.put(0, SIZE);
	        kernel2DGlobalWorkSize.put(1, SIZE);
	        kernel.setArg(0, aMem);
	        kernel.setArg(1, answerMem);
	        kernel.setArg(2, SIZE);
	        long Begin_gpuComp = System.nanoTime();
	        clEnqueueNDRangeKernel(queue, kernel, 2, null, kernel2DGlobalWorkSize, null, null, null);
	        long End_gpuComp = System.nanoTime();
	        
	        // Read the results memory back into our result buffer
	        clEnqueueReadBuffer(queue, answerMem, 1, 0, answer, null, null);
	        clFinish(queue);
	        double[][] GPU_Result = RMVG.BufferToMatrix(answer, SIZE);
	        long End_ResultTransfer = System.nanoTime();
	        
   	        // Clean up OpenCL resources
	        clReleaseKernel(kernel);
	        clReleaseProgram(program);
	        clReleaseMemObject(aMem);
	        clReleaseMemObject(answerMem);
	        clReleaseCommandQueue(queue);
	        clReleaseContext(context);
	        CL.destroy();     
	        long End_Full = System.nanoTime();
//	        System.out.println("GPU computation complete.");
	        
	        long Begin_cpuComp = System.nanoTime();
	        double[][] CPU_Result = CPUProcessing.transpose(A);
	        long End_cpuComp = System.nanoTime();
	        
//	        System.out.println("CPU computation complete.");
//	        
//	        if(RMVG.ResultsEqual(GPU_Result, CPU_Result)){
//	        	System.out.println("They Matched!\n");
//	        }
//	        
//	        System.out.println("\tTIME COMPARISONS:\n");
//	        System.out.println("GPU,FULL:\t" + (End_Full - Begin_Full));
//	        System.out.println("GPU,EXEC:\t" + (End_gpuComp - Begin_gpuComp));
//	        System.out.println("CPU,EXEC:\t" + (End_cpuComp - Begin_cpuComp));
	        
	        // Record results
	        retval.GPU_OpenCL_Overhead = (End_Init - Begin_Init) + (End_gpuComp - End_Full);
	        retval.GPU_DataTransfer = (Begin_gpuComp - Begin_DataTransfer) + (End_ResultTransfer - End_gpuComp);
	        retval.GPU_Exec = End_gpuComp - Begin_gpuComp;
	        retval.CPU_Exec = End_cpuComp - Begin_cpuComp;
	        
		} catch (LWJGLException e) {
			System.err.println("Looks like your system does not support OpenCL. :(");
			e.printStackTrace();
		} catch (IOException e){
			System.err.println("Oh no, there was an issue reading in your kernel!");
			e.printStackTrace();
		}
		
		return retval;
	}
	
	
	
	
	// For simplicity exception handling code is in the method calling this one.
	public static void initializeCL() throws LWJGLException { 
		IntBuffer errorBuff = BufferUtils.createIntBuffer(1);
		CL.create();
		platform = CLPlatform.getPlatforms().get(1); 
		devices = platform.getDevices(CL10.CL_DEVICE_TYPE_GPU);
		context = CLContext.create(platform, devices, errorBuff);
		queue = CL10.clCreateCommandQueue(context, devices.get(0), CL10.CL_QUEUE_PROFILING_ENABLE, errorBuff);
		Util.checkCLError(errorBuff.get(0)); 
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
