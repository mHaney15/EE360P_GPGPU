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
import java.nio.ByteBuffer;

import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;

import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLContext;
import org.lwjgl.opencl.CLDevice;
import org.lwjgl.opencl.CLPlatform;
import static org.lwjgl.opencl.CL10.*;

public class GPGPGU_XORENCRYPT_Driver {
	public static final int KEY = 42;
	public static CLContext context;
	public static CLPlatform platform;
	public static List<CLDevice> devices;
	public static CLCommandQueue queue;
	static ByteBuffer a;
	static ByteBuffer answer;
	static String kernelString; 
	
	
	public static void main(String[] args){
		try {
			
			System.out.println("Reading In a chapter of Pride and Prejudice...");
			String chapter = readFile("src/chap01");
			System.out.println("First 100 Characters:\n"+chapter.substring(0, 100));
			long Begin_Full = System.nanoTime();
			//Read in Kernel file.
			kernelString = readFile("src/Encrypt.cl");
			//System.out.println(kernelString);
			initializeCL();
			//Need to create charbuffer here!
			byte[] bytes = chapter.getBytes();
			a = BufferUtils.createByteBuffer(bytes.length);
			for(int i = 0; i < chapter.length(); i += 1) {
				a.put(bytes[i]);
			}
			
			String s = BufferToString(a);
			System.out.println(s);
			
			answer = BufferUtils.createByteBuffer(a.capacity());
			
			// Allocate memory for our two input buffers and our result buffer
			CLMem aMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a, null);
	        clEnqueueWriteBuffer(queue, aMem, 1, 0, a, null, null);
	        CLMem answerMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, answer, null);
	        clFinish(queue);
	        CLProgram program = clCreateProgramWithSource(context, kernelString, null);
	        //displayInfo();
	        clBuildProgram(program, devices.get(0), "", null);
	        System.out.println(program.getBuildInfoString(
	        devices.get(0), CL_PROGRAM_BUILD_LOG));
	        Util.checkCLError(clBuildProgram(program, devices.get(0), "", null));
	        // sum has to match a kernel method name in the OpenCL source
	        CLKernel kernel = clCreateKernel(program, "Encrypt", null);
	 
	        // Execution our kernel
	        PointerBuffer kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
	        kernel1DGlobalWorkSize.put(0, a.capacity());
	        
	        long Begin_gpuComp = System.nanoTime();
	        kernel.setArg(0, aMem);
	        kernel.setArg(1, answerMem);
	        kernel.setArg(2, KEY);
	        clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);
	        long End_gpuComp = System.nanoTime();
	        // Read the results memory back into our result buffer
	        clEnqueueReadBuffer(queue, answerMem, 1, 0, answer, null, null);
	        clFinish(queue);
	        String GPU_Result = BufferToString(answer);
	     // Clean up OpenCL resources
	        clReleaseKernel(kernel);
	        clReleaseProgram(program);
	        clReleaseMemObject(aMem);
	        clReleaseMemObject(answerMem);
	        clReleaseCommandQueue(queue);
	        clReleaseContext(context);
	        CL.destroy();     
	        long End_Full = System.nanoTime();
	        System.out.println("GPU encryption complete, first 100 characteres:");
	        System.out.println(GPU_Result.substring(0,100));
	        long Begin_cpuComp = System.nanoTime();
	        String CPU_Result = CPUProcessing.encrypt(GPU_Result, KEY);
	        String comp = CPUProcessing.encrypt(chapter, KEY);
	        
	        if(GPU_Result.equals(comp)){
	        	System.out.println("YAY");
	        } else {
	        	System.out.println("POOP");
	        }
	        
	        long End_cpuComp = System.nanoTime();
	        System.out.println("CPU decrpytion complete, first 100 characters:");
	        System.out.println(CPU_Result.substring(0,100));
	        
	        if(chapter.equals(CPU_Result)){
	        	System.out.println("They Matched!\n");
	        }
	        else{
	        	System.out.println("Uh-oh, something went wrong...");
	        }
	        
	        System.out.println("\tTIME COMPARISONS:\n");
	        System.out.println("GPU,FULL:\t" + (End_Full - Begin_Full));
	        System.out.println("GPU,EXEC:\t" + (End_gpuComp - Begin_gpuComp));
	        System.out.println("CPU,EXEC:\t" + (End_cpuComp - Begin_cpuComp));
	        
	        
	        
		} catch (LWJGLException e) {
			System.err.println("Looks like your system does not support OpenCL. :(");
			e.printStackTrace();
		} catch (IOException e){
			System.err.println("Oh no, there was an issue reading in your kernel!");
			e.printStackTrace();
		}
		
		
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
    
    public static String BufferToString(ByteBuffer BB){
    	byte[] bytes= new byte[BB.capacity()];
    	for(int i = 0; i < bytes.length; i++){
    		bytes[i] = BB.get(i);
    	}
    	return new String(bytes);
    }
}
