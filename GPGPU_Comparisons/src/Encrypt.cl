kernel void Encrypt( global char* EncStrng, global char* DecStrng, int Key)
{
	unsigned int xid = get_global_id(0);
	DecStrng[xid] = EncStrng[xid]^Key;
}
