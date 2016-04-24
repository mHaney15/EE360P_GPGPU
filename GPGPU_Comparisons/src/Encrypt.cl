kernel void Encrypt(global const char* DecStrng, global char* EncStrng, int Key)
{
	unsigned int xid = get_global_id(0);
	EncStrng[xid] = DecStrng[xid];
	//EncStrng[xid] = DecStrng[xid]^Key;
}
