//using Numpy;
using System;
using NumSharp;

public class CEM
{
    // Fixed parameters
    public int in_size = 5;    //Number of observations (pos_x, pos_y, theta, goal_x, goal_y)
    public int out_size = 2;  //Number of o

    //Policy parameters
    public int hidden_size = 5;    // How many values in the hidden layer
    public int evalation_samples = 1; // How many samples to take when evaluating a network

    //Training parameters
    public int cem_iterations = 100;    // How many total CEM iterations 
    public int cem_batch_size = 50;     // How many guassian samples in each CEM iteration
    public float cem_elite_frac = 0.5f;    // What percentage of cem samples are used to fit the guassian for next iteration
    public float cem_init_stddev = 1.0f;   // Initial CEM guassian uncertainty
    public float cem_noise_factor = 1.0f;    // Scaling factor of how much extra noise to add each iteration (noise_factor/iteration_number noise is added to std.dev.)
    public float cem_print_rate = 5;

    // Simulation paramters
    public float dt = 0.1f;    //seconds
    public int runtime = 8; //seconds



    //Car dynamics paramters
    public int v_max = 80;  //units/sec
    public float omega_max = 3.14f; //pi radians/sec = 180 deg/sec turn speed

    //Car shape
    public int car_w = 5;
    public int car_l = 10;

    //Target task
    NDArray car_start = np.array((-50, 0, 0.751));
    NDArray car_goal = np.array((50, 0));

    //int linear_policy_size = (in_size + 1) * out_size;
    public NDArray linear_model(NDArray param, NDArray in_data) {
        var in_vec = np.array(in_data).reshape(in_size, 1);
        var m1_end = in_size * out_size;
        var matrix1 = np.reshape(param["0:m1_end"], (out_size, in_size));
        var biases1 = np.reshape(param["m1_end: m1_end + out_size"], out_size);
        var result = np.matmul(matrix1, in_vec) + biases1;

        return result;
    }

    //cur_Model = linear_model
    //policy_size = linear_policy_size
    //two_layer_policy_size = (in_size+1)*hidden_size + (hidden_size+1)*out_size
    public NDArray two_layer_model(NDArray param, NDArray in_data) {
        //place input data in a column vector
        var in_vec = np.array(in_data).reshape(in_size, 1);

        //Layer 1 (input -> hidden)
        var m1_end = hidden_size * in_size;
        var matrix1 = np.reshape(param["0:m1_end"], (hidden_size, in_size));
        var biases1 = np.reshape(param["m1_end: m1_end + hidden_size"], (hidden_size, 1));
        var hidden_out = np.matmul(matrix1, in_vec) + biases1;
        hidden_out = np.matmul(hidden_out, (hidden_out > 0)) + 0.1 * np.matmul(hidden_out, (hidden_out < 0)); //Leaky ReLU

        //Layer 2 (hiden -> output);
        var m2_start = m1_end + hidden_size;
        var m2_end = m2_start + out_size * hidden_size;
        var matrix2 = np.reshape(param["m2_start: m2_end"], (out_size, hidden_size));
        var biases2 = np.reshape(param["m2_end: m2_end + out_size"], (out_size, 1));
        var result = np.matmul(matrix2, hidden_out) + biases2;
        result = result.reshape(out_size);
        return result;
    }

    //cur_Model = two_layer_model
    //policy_size = two_layer_policy_size
    public Tuple<double, double, double, double, double, double> cem(Func<double, double, double> f,
                                                                NDArray th_mean,
                                                                int batch_size,
                                                                int n_iter,
                                                                float elite_frac,
                                                                double initial_std = 1.0) {
        var n_elite = Math.Round(batch_size * elite_frac, 0);
        var th_std = np.ones_like(th_mean) * initial_std;
        NDArray ths, ys;
        for (int iter = 0; iter < n_iter; iter++) {
            foreach (double dth in th_std["None,:"])
            {
                ths = np.array(th_mean + dth * np.random.randn(batch_size, th_mean.size));
            }


        }
        //Add noise to batch_size samples 
        foreach (double dth in th_std["None,:"]) {
                ths = np.array(th_mean + dth * np.random.randn(batch_size, th_mean.size));
            }

            // Evaluate each sample
            foreach (double th in ths) {
                ys = np.array( [f(th, evalation_samples)] ) ; 
            }

            // Keep top n_elite best samples
            var elite_inds = ys.argsort()["::- 1"][":n_elite"];
            var elite_ths = ths[elite_inds];
            //Compute the mean and std-dev of best samples
            //var th_mean = elite_ths.Mean(axis = 0);
            //var th_std = elite_ths.std(axis = 0);
            var th_mean = elite_ths.mean();
            var th_std = np.std(elite_ths);
            //Add some extra noise
            var th_std = th_std + cem_noise_factor / (iter + 1);

            //Return results 
            return (ys : ys, theta_mean : th_mean, theta_std : th_std, y_mean : ys.mean(), f_th_mean : f(th_mean, 100), mean_of_std_dev: th_std.mean() );




    }










}




        
        
        

        

  

