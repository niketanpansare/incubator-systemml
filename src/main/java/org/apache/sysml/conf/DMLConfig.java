/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.conf;

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.LocalFileUtils;


public class DMLConfig 
{

	public static final String DEFAULT_SYSTEMML_CONFIG_FILEPATH = "./SystemML-config.xml";
	
	private static final Log LOG = LogFactory.getLog(DMLConfig.class.getName());
	
	// external names of configuration properties 
	// (single point of change for all internal refs)
	public static final String LOCAL_TMP_DIR        = "localtmpdir";
	public static final String SCRATCH_SPACE        = "scratch";
	public static final String OPTIMIZATION_LEVEL   = "optlevel";	
	public static final String NUM_REDUCERS         = "numreducers";
	public static final String JVM_REUSE            = "jvmreuse";
	public static final String DEFAULT_BLOCK_SIZE   = "defaultblocksize"; 	
	public static final String YARN_APPMASTER       = "dml.yarn.appmaster"; 	
	public static final String YARN_APPMASTERMEM    = "dml.yarn.appmaster.mem"; 
	public static final String YARN_MAPREDUCEMEM    = "dml.yarn.mapreduce.mem"; 
	public static final String YARN_APPQUEUE    	= "dml.yarn.app.queue"; 
	public static final String CP_PARALLEL_MATRIXMULT = "cp.parallel.matrixmult";
	public static final String CP_PARALLEL_TEXTIO   = "cp.parallel.textio";
	public static final String COMPRESSED_LINALG    = "compressed.linalg";
	// Fraction of available memory to use. The available memory is computer when the JCudaContext is created
	// to handle the tradeoff on calling cudaMemGetInfo too often.
	public static final String GPU_MEMORY_UTILIZATION_FACTOR    = "gpu.memory.util.factor";
	// Invoke cudaMemGetInfo to get available memory information. Useful if GPU is shared among multiple application. 
	public static final String REFRESH_AVAILABLE_MEMORY_EVERY_TIME    = "gpu.memory.refresh";

	// supported prefixes for custom map/reduce configurations
	public static final String PREFIX_MAPRED = "mapred";
	public static final String PREFIX_MAPREDUCE = "mapreduce";
	
	//internal config
	public static final String DEFAULT_SHARED_DIR_PERMISSION = "777"; //for local fs and DFS
	public static String LOCAL_MR_MODE_STAGING_DIR = null;
	
	//configuration default values
	private static HashMap<String, String> _defaultVals = null;

    private String _fileName = null;
	private Element _xmlRoot = null;
	
	static
	{
		_defaultVals = new HashMap<String, String>();
		_defaultVals.put(LOCAL_TMP_DIR,          "/tmp/systemml" );
		_defaultVals.put(SCRATCH_SPACE,          "scratch_space" );
		_defaultVals.put(OPTIMIZATION_LEVEL,     String.valueOf(OptimizerUtils.DEFAULT_OPTLEVEL.ordinal()) );
		_defaultVals.put(NUM_REDUCERS,           "10" );
		_defaultVals.put(JVM_REUSE,              "false" );
		_defaultVals.put(DEFAULT_BLOCK_SIZE,     String.valueOf(OptimizerUtils.DEFAULT_BLOCKSIZE) );
		_defaultVals.put(YARN_APPMASTER,         "false" );
		_defaultVals.put(YARN_APPMASTERMEM,      "2048" );
		_defaultVals.put(YARN_MAPREDUCEMEM,      "-1" );
		_defaultVals.put(YARN_APPQUEUE,    	     "default" );
		_defaultVals.put(CP_PARALLEL_MATRIXMULT, "true" );
		_defaultVals.put(CP_PARALLEL_TEXTIO,     "true" );
		_defaultVals.put(COMPRESSED_LINALG,      "false" );
		_defaultVals.put(GPU_MEMORY_UTILIZATION_FACTOR,      "0.9" );
		_defaultVals.put(REFRESH_AVAILABLE_MEMORY_EVERY_TIME,      "true" );
	}
	
	public DMLConfig()
	{
		
	}
	
	/**
	 * 
	 * @param fileName
	 * @throws ParseException
	 * @throws FileNotFoundException 
	 */
	public DMLConfig(String fileName) 
		throws ParseException, FileNotFoundException
	{
		this( fileName, false );
	}
	
	/**
	 * 
	 * @param fileName
	 * @param silent
	 * @throws ParseException
	 * @throws FileNotFoundException 
	 */
	public DMLConfig(String fileName, boolean silent) 
		throws ParseException, FileNotFoundException
	{
		_fileName = fileName;
		try {
			parseConfig();
		} catch (FileNotFoundException fnfe) {
			LOCAL_MR_MODE_STAGING_DIR = getTextValue(LOCAL_TMP_DIR) + "/hadoop/mapred/staging";
			throw fnfe;
		} catch (Exception e){
		    //log error, since signature of generated ParseException doesn't allow to pass it 
			if( !silent )
				LOG.error("Failed to parse DML config file ",e);
			throw new ParseException("ERROR: error parsing DMLConfig file " + fileName);
		}
		
		LOCAL_MR_MODE_STAGING_DIR = getTextValue(LOCAL_TMP_DIR) + "/hadoop/mapred/staging";
	}
	
	public DMLConfig( Element root )
	{
		_xmlRoot = root;
	}
	
	public void merge(DMLConfig otherConfig) 
		throws ParseException
	{
		if (otherConfig == null) 
			return;
	
		try {
			// for each element in otherConfig, either overwrite existing value OR add to defaultConfig
			NodeList otherConfigNodeList = otherConfig._xmlRoot.getChildNodes();
			if (otherConfigNodeList != null && otherConfigNodeList.getLength() > 0){
				for (int i=0; i<otherConfigNodeList.getLength(); i++){
					org.w3c.dom.Node optionalConfigNode = otherConfigNodeList.item(i);
					
					if (optionalConfigNode.getNodeType() == org.w3c.dom.Node.ELEMENT_NODE){
					
						// try to find optional config node in default config node
						String paramName = optionalConfigNode.getNodeName();
						String paramValue = ((Element)optionalConfigNode).getFirstChild().getNodeValue();
					
						if (_xmlRoot.getElementsByTagName(paramName) != null)
							LOG.info("Updating " + paramName + " with value " + paramValue);
						else 
							LOG.info("Defining new attribute" + paramName + " with value " + paramValue);
						DMLConfig.setTextValue(_xmlRoot, paramName, paramValue);
					}
					
				}
			} // end if (otherConfigNodeList != null && otherConfigNodeList.getLength() > 0){
		} catch (Exception e){
			LOG.error("Failed in merge default config file with optional config file",e);
			throw new ParseException("ERROR: error merging config file " + otherConfig._fileName + " with " + _fileName);
		}
	}
	
	/**
	 * Method to parse configuration
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	private void parseConfig () throws ParserConfigurationException, SAXException, IOException 
	{
		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		factory.setIgnoringComments(true); //ignore XML comments
		DocumentBuilder builder = factory.newDocumentBuilder();
		Document domTree = null;
		if (_fileName.startsWith("hdfs:") ||
		    _fileName.startsWith("gpfs:") )  // config file from DFS
		{
			if( !LocalFileUtils.validateExternalFilename(_fileName, true) )
				throw new IOException("Invalid (non-trustworthy) hdfs config filename.");
			FileSystem DFS = FileSystem.get(ConfigurationManager.getCachedJobConf());
            Path configFilePath = new Path(_fileName);
            domTree = builder.parse(DFS.open(configFilePath));  
		}
		else  // config from local file system
		{
			if( !LocalFileUtils.validateExternalFilename(_fileName, false) )
				throw new IOException("Invalid (non-trustworthy) local config filename.");
			domTree = builder.parse(_fileName);
		}
		
		_xmlRoot = domTree.getDocumentElement();		
	}
	
	/**
	 * Method to get string value of a configuration parameter
	 * Handles processing of configuration parameters 
	 * @param tagName the name of the DMLConfig parameter being retrieved
	 * @return a string representation of the DMLConfig parameter value.  
	 */
	public String getTextValue(String tagName) 
	{
		//get the actual value
		String retVal = (_xmlRoot!=null)?getTextValue(_xmlRoot,tagName):null;
		
		if (retVal == null)
		{
			if( _defaultVals.containsKey(tagName) )
				retVal = _defaultVals.get(tagName);
			else
				LOG.error("Error: requested dml configuration property '"+tagName+"' is invalid.");
		}
		
		return retVal;
	}
	
	public int getIntValue( String tagName )
	{
		return Integer.parseInt( getTextValue(tagName) );
	}
	
	public boolean getBooleanValue( String tagName )
	{
		return Boolean.parseBoolean( getTextValue(tagName) );
	}
	
	public double getDoubleValue( String tagName )
	{
		return Double.parseDouble( getTextValue(tagName) );
	}
	
	/**
	 * Method to get the string value of an element identified by tag
	 * @param ele
	 * @param tagName
	 * @return
	 */
	private static String getTextValue(Element element, String tagName) {
		String textVal = null;
		NodeList list = element.getElementsByTagName(tagName);
		if (list != null && list.getLength() > 0) {
			Element elem = (Element) list.item(0);
			textVal = elem.getFirstChild().getNodeValue();
			
		}
		return textVal;
	}
	
	/**
	 * Method to update the string value of an element identified by tagname 
	 * @param ele
	 * @param tagName
	 * @param newTextValue
	 */
	private static void setTextValue(Element element, String tagName, String newTextValue) {
		
		NodeList list = element.getElementsByTagName(tagName);
		if (list != null && list.getLength() > 0) {
			Element elem = (Element) list.item(0);
			elem.getFirstChild().setNodeValue(newTextValue);	
		}
	}
	
	/**
	 * Method to update the key value
	 * @param paramName
	 * @param paramValue
	 */
	public void setTextValue(String paramName, String paramValue) throws DMLRuntimeException {
		if(_xmlRoot != null)
			DMLConfig.setTextValue(_xmlRoot, paramName, paramValue);
		else {
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			factory.setIgnoringComments(true); //ignore XML comments
			DocumentBuilder builder;
			try {
				builder = factory.newDocumentBuilder();
				String configString = "<root><" + paramName + ">"+paramValue+"</" + paramName + "></root>";
				Document domTree = builder.parse(new ByteArrayInputStream(configString.getBytes("UTF-8")));
				_xmlRoot = domTree.getDocumentElement();
			} catch (Exception e) {
				throw new DMLRuntimeException("Unable to set config value", e);
			}
		}
	}

	/**
	 * Get a map of key/value pairs of all configurations w/ the prefix 'mapred'
	 * or 'mapreduce'. 
	 * 
	 * @return
	 */
	public Map<String, String> getCustomMRConfig()
	{
		HashMap<String, String> ret = new HashMap<String, String>();
	
		//check for non-existing config xml tree
		if( _xmlRoot == null )
			return ret;
		
		//get all mapred.* and mapreduce.* tag / value pairs		
		NodeList list = _xmlRoot.getElementsByTagName("*");
		for( int i=0; list!=null && i<list.getLength(); i++ ) {
			if( list.item(i) instanceof Element &&
				(  ((Element)list.item(i)).getNodeName().startsWith(PREFIX_MAPRED) 
				|| ((Element)list.item(i)).getNodeName().startsWith(PREFIX_MAPREDUCE)) )
			{
				Element elem = (Element) list.item(i);
				ret.put(elem.getNodeName(), 
						elem.getFirstChild().getNodeValue());
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 * @throws DMLRuntimeException
	 */
	public synchronized String serializeDMLConfig() 
		throws DMLRuntimeException
	{
		String ret = null;
		try
		{		
			Transformer transformer = TransformerFactory.newInstance().newTransformer();
			transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
			//transformer.setOutputProperty(OutputKeys.INDENT, "yes");
			StreamResult result = new StreamResult(new StringWriter());
			DOMSource source = new DOMSource(_xmlRoot);
			transformer.transform(source, result);
			ret = result.getWriter().toString();
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to serialize DML config.", ex);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param content
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static DMLConfig parseDMLConfig( String content ) 
		throws DMLRuntimeException
	{
		DMLConfig ret = null;
		try
		{
			//System.out.println(content);
			DocumentBuilder builder = DocumentBuilderFactory.newInstance().newDocumentBuilder();
			Document domTree = null;
			domTree = builder.parse( new ByteArrayInputStream(content.getBytes("utf-8")) );
			Element root = domTree.getDocumentElement();
			ret = new DMLConfig( root );
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to parse DML config.", ex);
		}
		
		return ret;
	}
	
	/**
	 * Start with the internal default settings, then merge in the
	 * settings from any specified configuration file, if available.
	 * If it is not explicitly given, then merge in settings from
	 * the default configuration file location, if available.
	 *
	 * @param configPath User-defined path of the configuration file.
	 * @return DMLConfig object containing configuration settings.
	 * @throws ParseException 
	 * @throws FileNotFoundException 
	 */
	public static DMLConfig readConfigurationFile(String configPath)
		throws ParseException, FileNotFoundException
	{
		// Always start with the internal defaults
		DMLConfig config = new DMLConfig();

		// Merge in any specified or default configs if available
		if (configPath != null) {
			// specified
			try {
				config = new DMLConfig(configPath, false);
			} catch (FileNotFoundException fnfe) {
				LOG.error("Custom config file " + configPath + " not found.");
				throw fnfe;
			} catch (ParseException e) {
				throw e;
			}
		} else {
			// default
			try {
				config = new DMLConfig(DEFAULT_SYSTEMML_CONFIG_FILEPATH, false);
			} catch (FileNotFoundException fnfe) {
				LOG.info("Using internal default configuration settings.  If you wish to " +
						 "customize any settings, please supply a `SystemML-config.xml` file.");
				config = new DMLConfig();
			} catch (ParseException e) {
				throw e;
			}
		}
		return config;
	}

	/**
	 * 
	 * @return
	 */
	public String getConfigInfo() 
	{
		String[] tmpConfig = new String[] { 
				LOCAL_TMP_DIR,SCRATCH_SPACE,OPTIMIZATION_LEVEL,
				NUM_REDUCERS, DEFAULT_BLOCK_SIZE,
				YARN_APPMASTER, YARN_APPMASTERMEM, YARN_MAPREDUCEMEM, 
				CP_PARALLEL_MATRIXMULT, CP_PARALLEL_TEXTIO
		}; 
		
		StringBuilder sb = new StringBuilder();
		for( String tmp : tmpConfig )
		{
			sb.append("INFO: ");
			sb.append(tmp);
			sb.append(": ");
			sb.append(getTextValue(tmp));
			sb.append("\n");
		}
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @param amMem
	 * @param mrMem
	 */
	public void updateYarnMemorySettings(String amMem, String mrMem)
	{
		//app master memory
		NodeList list1 = _xmlRoot.getElementsByTagName(YARN_APPMASTERMEM);
		if (list1 != null && list1.getLength() > 0) {
			Element elem = (Element) list1.item(0);
			elem.getFirstChild().setNodeValue(String.valueOf(amMem));
		}
		
		//mapreduce memory
		NodeList list2 = _xmlRoot.getElementsByTagName(YARN_MAPREDUCEMEM);
		if (list2 != null && list2.getLength() > 0) {
			Element elem = (Element) list2.item(0);
			elem.getFirstChild().setNodeValue(String.valueOf(mrMem));
		}
	}
	
	/**
	 * 
	 * @throws IOException
	 */
	@SuppressWarnings("deprecation")
	public void makeQualifiedScratchSpacePath() 
		throws IOException
	{
		NodeList list2 = _xmlRoot.getElementsByTagName(SCRATCH_SPACE);
		if (list2 != null && list2.getLength() > 0) {
			Element elem = (Element) list2.item(0);
			
			FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
			String fname = elem.getFirstChild().getNodeValue();
			Path path = new Path(fname).makeQualified(fs);
			
			elem.getFirstChild().setNodeValue(path.toString());
		}
	}
	
	/**
	 * 
	 * @param key
	 * @return
	 */
	public static String getDefaultTextValue( String key ) {
		return _defaultVals.get( key );
	}
	
	/**
	 * 
	 */
	public DMLConfig clone() {
		DMLConfig conf = new DMLConfig();
		conf._fileName = _fileName;
		conf._xmlRoot = (Element) _xmlRoot.cloneNode(true);
		
		return conf;
	}
}
