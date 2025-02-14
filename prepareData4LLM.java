
package ComSomeIp_NetMtrx_ums.KI;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import com.bosch.bct.javaaction.accessors.IContainer;
import com.bosch.bct.javaaction.accessors.IDefinition;
import com.bosch.bct.javaaction.accessors.IModule;
import com.bosch.bct.javaaction.accessors.IParameter;
import com.bosch.bct.javaaction.accessors.IReferenceParameter;
import com.bosch.bct.javaaction.context.interfaces.IContext;
import com.bosch.bct.javaaction.context.interfaces.IJavaAction;
import com.bosch.bct.javaaction.context.interfaces.ILog;

import ecucvalues.EcucValues;
import ecucvalues.Sd.SdConfig.SdInstance;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.*;
import java.util.*;

import java.util.*;

/*class Container {
    private Map<String, Object> data = new HashMap<>();

    public Container(Map<String, Object> initialData) {
        if (initialData != null) {
            data.putAll(initialData);
        }
    }

    public void addNestedContainer(String key, Container container) {
        data.put(key, container.getData());
    }

    public void addParameter(String key, Object value) {
        data.put(key, value);
    }

    public Object getParameter(String key) {
        return data.get(key);
    }

    public void setParameter(String key, Object value) {
        data.put(key, value);
    }

    public void addToArray(String key, Object value) {
        data.computeIfAbsent(key, k -> new ArrayList<>());
        List<Object> list = (List<Object>) data.get(key);
        if (value instanceof Container) {
            list.add(((Container) value).getData());
        } else {
            list.add(value);
        }
    }

    public Map<String, Object> getData() {
        return data;
    }
}
*/
import java.util.*;

import java.util.*;

class Container {
    private Map<String, Object> container;

    public Container(Map<String, Object> initialData) {
        if (initialData != null) {
            this.container = new HashMap<>(initialData);
        } else {
            this.container = new HashMap<>();
        }
    }

    public void addNestedContainer(String key, Container container) {
        this.container.put(key, container.getData());
    }

    public void addParameter(String key, Object value) {
        this.container.put(key, value);
    }

    public Object getParameter(String key) {
        return this.container.get(key);
    }

    public void setParameter(String key, Object value) {
        this.container.put(key, value);
    }

    public void addToArray(String key, Object value) {
        this.container.computeIfAbsent(key, k -> new ArrayList<>());
        List<Object> list = (List<Object>) this.container.get(key);
        if (value instanceof Container) {
            list.add(((Container) value).getData());
        } else {
            list.add(value);
        }
    }

    public Map<String, Object> getData() {
        return this.container;
    }
}

public class prepareData4LLM implements IJavaAction {
    Gson gson = new GsonBuilder().setPrettyPrinting().create();
	static Container root = null;
	
	public IDefinition printContainerInfo(EcucValues ecuc, IContainer container_name, ILog logger, String indentation, Container parentCont) {
	    Map<String, List<Object>> sortContainer = new HashMap<>();

	    if (container_name.getParameters().size() > 0)
	    {
			for (IParameter iparam : container_name.getParameters()) {
			/*	if (iparam instanceof IReferenceParameter)
				{
					parentCont.addParameter(iparam.getShortName(), ((IReferenceParameter) iparam).getTargetShortName());				
				}
				else
				{*/
					Container paramContainer = new Container(null);
					paramContainer.addParameter("value", iparam.getStringValue());
					paramContainer.addParameter("mandatory", isMandatory(iparam.getDefinition()));
					parentCont.addNestedContainer(iparam.getShortName(), paramContainer);
			//	}
			}
	    }
	    
		for (IContainer cont : container_name.getContainers()) {
			if (sortContainer.get(cont.getDefinition().getShortName()) == null)
			{
				ArrayList obj_lst = new ArrayList<>();
				obj_lst.add(cont);
				sortContainer.put(cont.getDefinition().getShortName(), obj_lst);
			}
			else
			{
				sortContainer.get(cont.getDefinition().getShortName()).add(cont);
			}
		};
		
		for (String contName : sortContainer.keySet()) {	
			if (sortContainer.get(contName).size() == 1)
			{
				Container contChild = new Container(null);
				parentCont.addNestedContainer(contName, contChild);
				contChild.addParameter("mandatory", isMandatory(((IContainer) sortContainer.get(contName).get(0)).getDefinition()));
				printContainerInfo(ecuc, (IContainer) sortContainer.get(contName).get(0), logger, indentation, contChild);
			}
			else
			{
				for (Object cont: sortContainer.get(contName))
				{
					Container contChild = new Container(null);
					parentCont.addToArray(contName, contChild);
					contChild.addParameter("mandatory", isMandatory(((IContainer) cont).getDefinition()));
					printContainerInfo(ecuc, (IContainer) cont, logger, indentation, contChild);
				}
			}
		}
	
		return null;
	}

	public static Long getMultiplicty(IDefinition definition, String whichOne) {

		if (definition == null) {
			return null;
		}

		switch (whichOne.toLowerCase()) {
		case "upper":
			return definition.getUpperMultiplicity();

		case "lower":
			return definition.getLowerMultiplicity();

		default:
			return null;
		}
	}
	
	public static String isMandatory(IDefinition definition) {
        Long mandatory = getMultiplicty(definition, "lower");
        if (mandatory > 0)
        {
        	return String.valueOf(true);
		}
        return String.valueOf(false);
	}

	public void createModuleInfo(EcucValues ecuc, ILog logger) {
		List<IModule> module_lst = ecuc.getModules();
     	root = new Container(null);
		for (IModule module :module_lst)
		{
			if (module.getShortName().compareTo("Sd") == 0)
			{   
		        Container contChildModule = new Container(null);
		        root.addNestedContainer(module.getShortName(), contChildModule);
		        contChildModule.addParameter("mandatory", isMandatory(module.getDefinition()));
		        
			    module.getContainers().stream().forEach(e -> {
			    	Container contChild = new Container(null);
			    	contChildModule.addNestedContainer(e.getShortName(), contChild);
			    	contChild.addParameter("mandatory", isMandatory(e.getDefinition()));
			    	printContainerInfo(ecuc, e, logger, " ", contChild);
				});	
			}
		}
	}	
	
	@Override
	public void run(IContext context) {
		// Hashmap swc from Ford from csv
		String inputFile4LLMProcessing = prepareData4LLM.class.getProtectionDomain().getCodeSource().getLocation().getPath()
				.replace("/.bin", "") + "ComVeh/ComSomeIp_NetMtrx/ComSomeIp_NetMtrx_ums/KI/_out/";
	
		EcucValues ecuc = context.getModel(EcucValues.class);
		ILog logger = context.getLog();
		//root = new Container("Cubas");
		createModuleInfo(ecuc, logger);

/*		File file = new File(inputFile4LLMProcessing + "inputFile4LLMProcessing.txt");
		FileWriter myWriter;
		try {
			myWriter = new FileWriter(file);
			myWriter.write(sResult.toString());
			myWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
*/
		
        String jsonOutput = gson.toJson(root.getData());
       // logger.info(jsonOutput);
        
        File file = new File(inputFile4LLMProcessing + "inputFile4LLMProcessing.json");
		FileWriter myWriter;
		try {
			myWriter = new FileWriter(file);
			myWriter.write(jsonOutput);
			myWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
/*	      Map<String, Object> rootData = new HashMap<>();
	        rootData.put("CustomTag", "MyCustomTag");
	        rootData.put("InstanceName", "MainInstance");
	        rootData.put("Description", "This is a dynamic container");

	        Container rootContainer = new Container(rootData);

	        // Fügt weitere dynamische Daten hinzu
	        rootContainer.addParameter("Version", "1.0.0");
	        rootContainer.addParameter("Enabled", true);

	        // Erstellt einen Untercontainer mit eigener Struktur
	        Map<String, Object> subData = new HashMap<>();
	        subData.put("ServiceID", "Service_1");
	        subData.put("ServiceInterface", "SomeServiceInterface");
	        subData.put("TransportProtocol", "SOME/IP");

	        Container subContainer = new Container(subData);

	        // IP-Konfiguration als verschachtelte Struktur
	        Map<String, Object> ipConfig = new HashMap<>();
	        ipConfig.put("IPAddress", "192.168.1.100");
	        ipConfig.put("Port", 30509);
	        subContainer.addParameter("IPConfiguration", ipConfig);

	        // Fügt den Untercontainer mit einem beliebigen Schlüssel hinzu
	        rootContainer.addNestedContainer("Sd", subContainer);
	        String jsonOutput = gson.toJson(rootContainer);
	        */

/*
        JsonBuilder example = new JsonBuilder();

        // Ein Beispiel-JsonObject erstellen
        JsonObject settings = new JsonObject();
        settings.addProperty("theme", "dark");

        example.addObject("settings", settings);
        
        
        // Vorheriges JSON ausgeben
        logger.info("Vorher:");
        logger.info(example.toJson());

        // Das Objekt 'language' holen oder erstellen
        JsonObject languageObject = example.getOrCreateObject("language", logger);

        // Einen neuen Eintrag in das 'language' Objekt hinzufügen
        languageObject.addProperty("default", "deutsch");

        // Nachheriges JSON ausgeben
        logger.info("Nachher:");
        logger.info(example.toJson());
        */
        
     /*   
        File file = new File(inputFile4LLMProcessing + "inputFile4LLMProcessing.json");
		FileWriter myWriter;
		try {
			myWriter = new FileWriter(file);
			myWriter.write(jsonOutput);
			myWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
	*/
	}
}
