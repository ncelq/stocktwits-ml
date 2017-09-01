package com.spark.poc.ml;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RequestMapping("api")
@RestController
public class ApiController {
	
    @Autowired
    MLModel mlModel;
    
    @RequestMapping("predict")
    public String predict( @RequestParam(value="text", defaultValue="up") String text) {
        return mlModel.predict(text);
    }
    
}
