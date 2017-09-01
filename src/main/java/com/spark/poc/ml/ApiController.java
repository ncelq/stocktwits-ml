package com.spark.poc.ml;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RequestMapping("api")
@RestController
public class ApiController {
	
    @Autowired
    MLModel mlModel;
    /*
    @RequestMapping("test")
    public void trigger() {
    	tfidf.trigger();
        return;
    }*/
    
}
