more off
clear

#---TASK 3---#

#load data 
data = csvread("datafile.csv");

#replace 11th attribute
data(data(:,11) == 2,11) = -1;
data(data(:,11) == 4,11) = 1;

#stored unique ID for later reference
id = data(:,1); 

#remove the 1st attribute 
data(:,1) = [];

#input data matrix type
X = data( : , 1:9);
Y = data( : , 10);

#---TASK 4---#

#seperate data into training and testing

#training 2/3
XTrain = X(1 : 466,:);
YTrain = Y(1 : 466,:);

#testing 1/3
XTest = X(467 : 699,:);
YTest = Y(467 : 699,:);

fid = fopen ("predictions.txt", "w");
 
#Intialize zero matrix
W = zeros(1,9);

#INPUT: aggressiveness parameter C > 0
C = 1;

#for all three variations of the PA algorithm
for algorithm = 1:3
  switch (algorithm)
    case 1
      fprintf ("\n#----------------------------PA ALGORITM----------------------------#\n");        
    case 2
      fprintf ("\n#---------------------------PA-I ALGORITM---------------------------#\n");
    case 3
      fprintf ("\n#---------------------------PA-II ALGORITM--------------------------#\n");
  endswitch
  
  #for 1, 2 and 10 iterations
  for iter = [1, 2, 10]
  
    switch (algorithm)
      case 1
        prediction_desc = sprintf('\nPredictions for PA Algorithm for number of iterations = %d',iter); 
        fdisp(fid, prediction_desc);
      case 2
        prediction_desc = sprintf('\nPredictions for PA-I Algorithm for number of iterations = %d',iter); 
        fdisp(fid, prediction_desc);    
      case 3
        prediction_desc = sprintf('\nPredictions for PA-II Algorithm for number of iterations = %d',iter); 
        fdisp(fid, prediction_desc);
    endswitch
    
    #calculate W for iter iteration
    for k = 1:iter
      
      #TRAINING for the first 2/3 of data set. j refers to a single row of the data set.
      for j = 1:466
 
        #receive instance: xt ∈ Rn (training set)
        xt = XTrain(j,:);
        
        #predict y_hat
        y_hat=sign(W * xt');
        
        #Recieve correct label yt
        yt = YTrain(j);
        
        training_results(j) = y_hat*yt;
        
        #suffer loss
        lt = max(0 , 1 - yt*(W*xt'));              
        
        #calculate torque for PA, PA I, PA II    
        switch (algorithm)
          case 1
            torque = lt / (norm(xt)^2);
          case 2
            torque = min(C, lt / (norm(xt)^2));
          case 3
            torque = lt / ((norm(xt)^2) + 1/(2*C));
        endswitch
        
        #update W 
        W = W + torque*yt*xt;
        
      end #for 1:466 
 
    end #for 1:iter
   
    #print Training output
    
    fprintf("\nNUMBER OF ITERATIONS = %d\n\n", iter) 
    
    W
    
    disp("#-------------Training Results-------------#");
    
    train_correct_predictions = sum(training_results == 1);
    fprintf("\nThe number of correct preditions made is %d\n", train_correct_predictions)
    
    train_incorrect_predictions = sum(training_results == -1);
    fprintf("The number of incorrect preditions made is %d\n", train_incorrect_predictions)
    
    training_accuracy = (train_correct_predictions / 466) * 100;
    fprintf("The training accuracy is %d%%\n\n", training_accuracy)    

    disp("#--------------Testing Results--------------#");
    
    #TESTING for the last 1/3 of data set. 
    
    #store results 
    testing_results = zeros([233,1]);

    #calculate test set accuracy 
    for t = 1:233
      id_test = id(467 : 699,:); 
      xt = XTest(t,:)'; 
      y_hat = sign(W * xt);
      yt = YTest(t);
      testing_results(t) = y_hat * yt;
      
      #print to predictions.txt --> correct result
      if (testing_results(t) == 1)
        fdisp(fid,strcat("Id =",num2str(id_test(t,1)),", y_hat =",num2str(y_hat),", yt =",num2str(yt),", y_hat*yt =",num2str(testing_results(t)),", "," CORRECT"));
      
      #print to predictions.txt --> incorrect result
      elseif (testing_results(t) == -1)
        fdisp(fid,strcat("Id =",num2str(id_test(t,1)),", y_hat =",num2str(y_hat),", yt =",num2str(yt),", y_hat*yt =",num2str(testing_results(t)),", "," INCORRECT"));
      endif
      
    end;
        
    #print Testing output
    
    test_correct_predictions = sum(testing_results == 1);
    fprintf("\nThe number of correct preditions made is %d\n", test_correct_predictions)
    
    test_incorrect_predictions = sum(testing_results == -1);
    fprintf("The number of incorrect preditions made is %d\n", test_incorrect_predictions)   
    
    testing_accuracy = (test_correct_predictions / 233) * 100;
    fprintf("The testing accuracy is %d%%\n\n", testing_accuracy)
    
  end #for i =[1,2,10]
  
end #for algorithm=1:3
  
fclose (fid);
    
    
    
  
