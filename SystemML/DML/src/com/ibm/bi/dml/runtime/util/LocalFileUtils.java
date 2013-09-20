package com.ibm.bi.dml.runtime.util;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

public class LocalFileUtils 
{
	public static final int BUFFER_SIZE = 8192;
	
	//unique IDs per JVM for tmp files
	private static IDSequence _seq = null;
	private static String _workingDir = null;
	
	//categories of temp files under process-specific working dir
	public static final String CATEGORY_CACHE        = "cache";
	public static final String CATEGORY_PARTITIONING = "partitioning";
	public static final String CATEGORY_RESULTMERGE  = "resultmerge";
	public static final String CATEGORY_WORK         = "work";
	
	static
	{
		_seq = new IDSequence();
	}
	
	/**
	 * 
	 * @param filePathAndName
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixBlockFromLocal(String filePathAndName)
		throws FileNotFoundException, IOException
	{
		return readMatrixBlockFromLocal(filePathAndName, new MatrixBlock());
	}
	
	/**
	 * 
	 * @param filePathAndName
	 * @return
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static MatrixBlock readMatrixBlockFromLocal(String filePathAndName, MatrixBlock ret)
		throws FileNotFoundException, IOException
	{
		FileInputStream fis = new FileInputStream( filePathAndName );
		BufferedInputStream bis = new BufferedInputStream( fis, BUFFER_SIZE );
		DataInputStream in = new DataInputStream( bis );
	
		try
		{
			ret.readFields(in);
		}
		finally
		{
			if( in != null )
				in.close();
		}
			
		return ret;
	}
	
	/**
	 * 
	 * @param filePathAndName
	 * @param mb
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static void writeMatrixBlockToLocal (String filePathAndName, MatrixBlock mb)
		throws FileNotFoundException, IOException
	{		
		FileOutputStream fos = new FileOutputStream( filePathAndName );
		//BufferedOutputStream bos = new BufferedOutputStream( fos, BUFFER_SIZE );
		//DataOutputStream out = new DataOutputStream( bos );
		FastBufferedDataOutputStream out = new FastBufferedDataOutputStream(fos, BUFFER_SIZE);
		
		try 
		{
			mb.write(out);
		}
		finally
		{
			if( out != null )
				out.close ();	
		}	
	}
	
	
	/**
	 * 
	 * @param filePathAndName
	 * @param data
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static void writeByteArrayToLocal (String filePathAndName, byte[] data)
		throws FileNotFoundException, IOException
	{		
		FileOutputStream fos = new FileOutputStream( filePathAndName );
		
		try 
		{
			fos.write(data);
		}
		finally
		{
			if( fos != null )
				fos.close ();	
		}	
	}

	
	/**
	 * NOTE: this is an experimental method (achieves 25% improvement for dense matrices)
	 * and is currently only applicable for dense matrices.
	 * 
	 * @param filePathAndName
	 * @param mb
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	@Deprecated 
	public static void writeMatrixBlockToLocal2(String filePathAndName, MatrixBlock mb)
		throws FileNotFoundException, IOException
	{		
		double[] a = mb.getDenseArray();
		int m = mb.getNumRows();
		int n = mb.getNumColumns();
		int BUFFER_NUM_VALS = BUFFER_SIZE/8;
		
		FileOutputStream fos = new FileOutputStream( filePathAndName );
		FileChannel file = fos.getChannel();
		ByteBuffer buf = ByteBuffer.allocate(BUFFER_SIZE);
		
		try 
		{
			buf.putInt(m);
			buf.putInt(n);
			file.write(buf);
			buf.clear();
			
			int limit = m*n;
			for(int i=0; i<limit; i+=BUFFER_NUM_VALS)
			{
				for( int j=0; j<BUFFER_NUM_VALS && i+j <limit; j++ )
					buf.putDouble(a[i+j]);
				
				file.write(buf);
				buf.clear();
			}
		}
		finally
		{
			if( file != null )
				file.close ();	
			if( fos != null )
				fos.close();
		}	
	}


	/**
	 * 
	 * @param filePathAndName
	 * @param outValues
	 * @return
	 * @throws IOException 
	 */
	public static int readBlockSequenceFromLocal( String filePathAndName, Pair<MatrixIndexes,MatrixValue>[] outValues, HashMap<MatrixIndexes, Integer> outMap) 
		throws IOException
	{
		FileInputStream fis = new FileInputStream( filePathAndName );
		BufferedInputStream bis = new BufferedInputStream( fis, BUFFER_SIZE );
		DataInputStream in = new DataInputStream( bis );
		int bufferSize = 0;
		
		try
		{
			int len = in.readInt();
			for( int i=0; i<len; i++ )
			{
				outValues[i].getKey().readFields(in);
				outValues[i].getValue().readFields(in);
				outMap.put( outValues[i].getKey(), i );
			}
			bufferSize = len;
		}
		finally
		{
			if( in != null )
				in.close();
		}
			
		return bufferSize;
	}
	
	/**
	 * 
	 * @param filePathAndName
	 * @param inValues
	 * @param len
	 * @throws IOException 
	 */
	public static void writeBlockSequenceToLocal( String filePathAndName, Pair<MatrixIndexes,MatrixValue>[] inValues, int len ) 
		throws IOException
	{
		FileOutputStream fos = new FileOutputStream( filePathAndName );
		FastBufferedDataOutputStream out = new FastBufferedDataOutputStream(fos, BUFFER_SIZE);
		
		try 
		{
			out.writeInt(len);
			for( int i=0; i<len; i++ )
			{
				inValues[i].getKey().write(out);
				inValues[i].getValue().write(out);
			}
		}
		finally
		{
			if( out != null )
				out.close ();	
		}	
	}


	/**
	 * 
	 * @param dir
	 */
	public static void createLocalFileIfNotExist(String dir) 
	{
		File fdir = new File(dir);
		if( !fdir.exists() )
		{
			fdir.mkdirs();
		}
	}
	
	/**
	 * 
	 * @param dir
	 */
	public static void deleteFileIfExists(String dir) 
	{
		File fdir = new File(dir);
		if( fdir.exists() )
			rDelete(fdir);
	}

	
	/**
	 * 
	 * @param dir
	 * @param permission
	 */
	public static void createLocalFileIfNotExist( String dir, String permission )
	{
		File fdir = new File(dir);
		if( !fdir.exists() )
		{
			fdir.mkdirs();
			setLocalFilePermissions(fdir, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		}
	}
	
	/**
	 * 
	 * @param file
	 * @param permissions
	 */
	public static void setLocalFilePermissions( File file, String permissions )
	{
		//note: user and group treated the same way
		char[] c = permissions.toCharArray();
		short sU = (short)(c[0]-48);
		short sO = (short)(c[2]-48); 
		
		file.setExecutable( (sU&1)==1, (sO&1)==0 );
		file.setWritable(   (sU&2)==2, (sO&2)==0 );
		file.setReadable(   (sU&4)==4, (sO&4)==0 );
	}

	
	///////////
	// working dir handling
	///
	
	/**
	 * 
	 * @param dir
	 * @return
	 */
	public static String checkAndCreateStagingDir(String dir) 
	{
		File f =  new File(dir);		
		if( !f.exists() )
			f.mkdirs();
		
		return dir;
	}

	/**
	 * 
	 * @return
	 */
	public static String createWorkingDirectory() 
	{
		return createWorkingDirectoryWithUUID( DMLScript.getUUID() );
	}
	
	/**
	 * 
	 * @return
	 */
	public static String createWorkingDirectoryWithUUID( String uuid ) 
	{
		//create local tmp dir if not existing
		String dirRoot = null;
		DMLConfig conf = ConfigurationManager.getConfig();
		if( conf != null ) 
			dirRoot = conf.getTextValue(DMLConfig.LOCAL_TMP_DIR);
		else 
			dirRoot = DMLConfig.getDefaultTextValue(DMLConfig.LOCAL_TMP_DIR);		
		//create shared staging dir if not existing
		LocalFileUtils.createLocalFileIfNotExist(dirRoot, DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
		
		//create process specific sub tmp dir
		StringBuilder sb = new StringBuilder();
		sb.append( dirRoot );
		sb.append(Lops.FILE_SEPARATOR);
		sb.append(Lops.PROCESS_PREFIX);
		sb.append( uuid );
		sb.append(Lops.FILE_SEPARATOR);
		_workingDir = sb.toString();
		
		//create process-specific staging dir if not existing
		LocalFileUtils.createLocalFileIfNotExist(_workingDir);
		
		return _workingDir;
	}
	
	/**
	 * 
	 */
	public static void cleanupWorkingDirectory() 
	{
		if( _workingDir != null )
			cleanupWorkingDirectory( _workingDir );
	}
		
	/**
	 * 
	 * @param dir
	 * @return
	 */
	public static void cleanupWorkingDirectory(String dir) 
	{
		File f =  new File(dir);
		if( f.exists() )
			rDelete(f);
	}
	
	/**
	 * 
	 * @param dir
	 */
	public static void rDelete(File dir)
	{
		//recursively delete files if required
		if( dir.isDirectory() )
		{
			File[] files = dir.listFiles();
			for( File f : files )
				rDelete( f );	
		}
		
		//delete file itself
		dir.delete();
	}

	/**
	 * 
	 * @return
	 */
	public static String getWorkingDir()
	{
		if( _workingDir == null )
			createWorkingDirectory();
		return _workingDir;
	}
	
	/**
	 * 
	 * @param category
	 * @return
	 */
	public static String getWorkingDir( String category )
	{
		if( _workingDir == null )
			createWorkingDirectory();
		
		StringBuilder sb = new StringBuilder();
		sb.append( _workingDir );
		sb.append( Lops.FILE_SEPARATOR );
		sb.append( category );
		sb.append( Lops.FILE_SEPARATOR );
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @param category
	 * @return
	 */
	public static String getUniqueWorkingDir( String category )
	{
		if( _workingDir == null )
			createWorkingDirectory();
		
		StringBuilder sb = new StringBuilder();
		sb.append( _workingDir );
		sb.append( Lops.FILE_SEPARATOR );
		sb.append( category );
		sb.append( Lops.FILE_SEPARATOR );
		sb.append( "tmp" );
		sb.append( _seq.getNextID() );
		
		return sb.toString();
	}
	
}
