$(function() {
	var tangle = new Tangle($('#pid-demo')[0], {
    	initialize: function () { 
    		this.KP = 1.0;
    		this.KI = 0.0;
    		this.KD = 1.0;
    	},
    	update: function () { 
    		drawChart();
    	}
	});


	function PID(KP, KI, KD) {
		this.feedback = [];
		this.integ = 0;
		this.prev_error = 0;

		this.set_feedback = function(value) {
			this.feedback.unshift(value);
			if(this.feedback.length > 10) {
				this.feedback.pop();
			}
		};

		this.get_feedback = function(i) {
			if(i >= this.feedback.length) return 0;
			return this.feedback[i];
		};

		this.act = function(target) {
			var error = target - this.get_feedback(0);
			var deriv = error - this.prev_error;
			this.integ = 0.1 * this.integ + error;

			this.prev_error = error;
			return KP * error + KD * deriv + KI * this.integ;
		};
	}
    

	function drawChart() {
		var N = 200;
		var pid = null;

		var val = -1;
		function newTarget(i) {
			if(i % 50 === 0) val *= -1;
			return val;
		}

		if(tangle) {
			pid = new PID(
				tangle.getValue('KP'),
				tangle.getValue('KI'),
				tangle.getValue('KD')
			);
		} else {
			pid   = new PID(1, 0, 1);
		}

		var data_ = [["Step", "Feedback", "Target"]];

		for(var i = 0; i < N; ++i) {
			var target = newTarget(i);
			controller_output = pid.act(target);

			var value = 2 * pid.get_feedback(0) - 1 * pid.get_feedback(1) + 0.1 * controller_output;
			
			pid.set_feedback(value);
			
			data_.push([i, value, target]);
		}

		var data = google.visualization.arrayToDataTable(data_);


		var options = {
		  backgroundColor: { fill:'transparent' },
		  legend : { position : 'none'},
		  tooltip : { trigger : 'none'},
		   'chartArea': {'width': '100%', 'height': '100%'}
		};

		var chart = new google.visualization.LineChart(document.getElementById('pid-chart'));
		chart.draw(data, options);
	}

	$(window).resize(drawChart);
	drawChart();
});