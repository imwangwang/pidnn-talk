$(function() {

	var Settings = function() {
		this.dt 	= 0.01;
		this.target = 0.08;
		this.M 		= 50;
	};

 	var settings = new Settings();


	function zeroArray(len) {
	    var rv = new Array(len);
	    while (--len >= 0) {
	        rv[len] = 0;
	    }
	    return rv;
	}

	function Neuron(memory, transfer_func) {
		this.input  = zeroArray(memory);
		this.output = zeroArray(memory);

		this.act = function(new_input) {
			var new_output = transfer_func.call(this, new_input);
		
			this.input.unshift(new_input);
			this.input.pop();
			
			this.output.unshift(new_output);
			this.output.pop();
			
			return new_output;
		};
	}

	function PIDNN(memory) {
		var transfer_X = function (x) {
			if(x > 1)  x =  1;
			if(x < -1) x = -1;
			return x;
		};
		
		var transfer_P = function(x) {
			if(x > 1)  x =  1;
			if(x < -1) x = -1;
			return x;
		};

		var transfer_I = function(x) {
			x = this.output[0];
			if(x > 1)  x =  1;
			if(x < -1) x = -1;
			return x;
		};

		var transfer_D = function(x) {
			x = x - this.input[0];
			if(x > 1)  x =  1;
			if(x < -1) x = -1;
			return x;
		};

		this.input_layer = [
			new Neuron(memory, transfer_X),
			new Neuron(memory, transfer_X)
		];

		this.input_weights = [
			[1, -1], [1, -1], [1, -1]
		];

		this.hidden_layer = [
			new Neuron(memory, transfer_P),
			new Neuron(memory, transfer_I),
			new Neuron(memory, transfer_D)
		];

		this.hidden_weights = [
			[1, 1, 10]
		];

		this.output_layer = [
			new Neuron(memory, transfer_X)
		];

		this.memory = memory;

		this.system_feedback = zeroArray(memory);

		this.act = function(target) {
			this.input_layer[0].act( target );
			this.input_layer[1].act( this.system_feedback[0] );

			for(var i = 0; i < this.hidden_layer.length; ++i) {
				var input = 0;
				for(var j = 0; j < this.input_layer.length; ++j) {
					input += this.input_layer[j].output[0] * this.input_weights[i][j];
				}

				this.hidden_layer[i].act(input);
			}

			for(var i = 0; i < this.output_layer.length; ++i) {
				var input = 0;
				for(var j = 0; j < this.hidden_layer.length; ++j) {
					input += this.hidden_layer[j].output[0] * this.hidden_weights[i][j];
				}

				this.output_layer[i].act(input);
			}	

			return this.output_layer[0].output[0];
		};

		this.update_feedback = function(value) {
			this.system_feedback.unshift(value);
			this.system_feedback.pop();
		};


		this.train = function() {
			var dt = settings.dt;
			var m  = this.memory - 1;

			function dz(x) {
				if(x === 0) return 0;
				return 1/x;
			}

			function sign(x) {
				if(x < 0) return -1;
				if(x > 0) return 1;
				return 0;
			}

			for(var i = 0; i < this.hidden_layer.length; ++i) {
				var delta = 0;

				for(var k = 0; k < m; ++k) {
					delta += 
						(this.input_layer[0].input[k] - this.input_layer[1].input[k])
						*sign((this.system_feedback[k] - this.system_feedback[k+1])
						*dz(this.output_layer[0].output[k] - this.output_layer[0].output[k+1]))
						*this.hidden_layer[i].output[k];	

				}

				delta *= (-2 / m);

				this.hidden_weights[0][i] -= dt * delta;
			}


			for(var i = 0; i < this.hidden_layer.length; ++i) {
				for(var j = 0; j < this.input_layer.length; ++j) {
					var delta = 0;

					for(var k = 0; k < m; ++k) {
						delta += 
						(this.input_layer[0].input[k] - this.input_layer[1].input[k])
						*sign((this.system_feedback[k] - this.system_feedback[k+1])
						*dz(this.output_layer[0].output[k] - this.output_layer[0].output[k+1]))
						*this.hidden_weights[0][i]
						*sign(
							(this.hidden_layer[i].output[k] - this.hidden_layer[i].output[k+1])
							*dz(this.hidden_layer[i].input[k] - this.hidden_layer[i].input[k+1])
						)
						*this.input_layer[j].output[k];


					}

					delta *= (-2 / m);

					this.input_weights[i][j] -= dt*dt * delta;
				}
			}

		};


	}



	function PID(memory) {
		this.system_feedback = zeroArray(memory);
		this.error 			 = zeroArray(memory);

		this.integ = 0;

		this.act = function(target) {
			
			var error = target - this.system_feedback[0];
			var deriv = error - this.error[0];

			this.integ = this.integ + error;

			this.update_error( error );
			


			var KP = 1;
			var KD = 1;
			var KI = 0;
			return KP * error + KD * deriv + KI * error;
		};


		this.update_feedback = function(value) {
			this.system_feedback.unshift(value);
			this.system_feedback.pop();
		};

		this.update_error = function(value) {
			this.error.unshift(value);
			this.error.pop();
		};

		this.train = function() {

		};	
	}


	function drawChart() {
		var M = settings.M;
		var N = 500;

		var pid   = new PID(M);
		var pidnn = new PIDNN(M);

		var controller_output1 = 0;
		var controller_output2 = 0;


		var data_ = [["Step", "PID controller", "PIDNN controller", "Target Value"]];

		for(var i = 0; i < N; ++i) {


			controller_output1 = pid.act(settings.target);
			controller_output2 = pidnn.act(settings.target);
			

			var value1 = 2 * pid.system_feedback[0] - 1 * pid.system_feedback[1] + 0.01 * controller_output1;
			var value2 = 2 * pidnn.system_feedback[0] - 1 * pidnn.system_feedback[1] + 0.01 * controller_output2;
			
			pid.update_feedback(value1);
			pidnn.update_feedback(value2);
			

			pidnn.train();
			
			data_.push([i, value1, value2, settings.target]);
		}

		var data = google.visualization.arrayToDataTable(data_);


		var options = {
		  backgroundColor: { fill:'transparent' },
		 	legend : { position : 'bottom'},
		  tooltip : { trigger : 'none'},
		   'chartArea': {'width': '80%', 'height': '80%'}
		};

		var chart = new google.visualization.LineChart(document.getElementById('pidnn-chart'));
		chart.draw(data, options);
	}

	$(window).resize(drawChart);
	drawChart();
});