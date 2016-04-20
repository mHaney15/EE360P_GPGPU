package GPGPU_Comparisons;

import java.nio.IntBuffer;
import java.util.List;

import org.lwjgl.BufferUtils;
import org.lwjgl.LWJGLException;
import org.lwjgl.opencl.*;

public class GPGPGU_Driver {
	public static CLContext context;
	public static CLPlatform platform;
	public static List<CLDevice> devices;
	public static CLCommandQueue queue;

	public static void main(String[] args){
		try {
			initializeCL();
		} catch (LWJGLException e) {
			System.err.println("Looks like your system does not support OpenCL. :(");
			e.printStackTrace();
		}
		
	}
	
	
	
	
	// For simplicity exception handling code is in the method calling this one.
	public static void initializeCL() throws LWJGLException { 
		IntBuffer errorBuff = BufferUtils.createIntBuffer(1);
		CL.create();
		platform = CLPlatform.getPlatforms().get(0); 
		devices = platform.getDevices(CL10.CL_DEVICE_TYPE_GPU);
		context = CLContext.create(platform, devices, errorBuff);
		queue = CL10.clCreateCommandQueue(context, devices.get(0), CL10.CL_QUEUE_PROFILING_ENABLE, errorBuff);
		Util.checkCLError(errorBuff.get(0)); 
	}	
}
