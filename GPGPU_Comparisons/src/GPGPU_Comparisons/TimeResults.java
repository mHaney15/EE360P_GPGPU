package GPGPU_Comparisons;
// Houses the timing data for test results

public class TimeResults {
	public long CPU_Exec = 0;
	public long GPU_Exec = 0;
	public long GPU_OpenCL_Overhead = 0;
	public long GPU_DataTransfer = 0;
	
	public void add(TimeResults other) {
		this.CPU_Exec += other.CPU_Exec;
		this.GPU_DataTransfer += other.GPU_DataTransfer;
		this.GPU_Exec += other.GPU_Exec;
		this.GPU_OpenCL_Overhead += other.GPU_OpenCL_Overhead;
	}
	
	public void divideAll(int div) {
		this.CPU_Exec /= div;
		this.GPU_DataTransfer /= div;
		this.GPU_Exec /= div;
		this.GPU_OpenCL_Overhead /= div;
	}
}
