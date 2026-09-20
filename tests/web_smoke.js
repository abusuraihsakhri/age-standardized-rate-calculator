"use strict";
const core=require("../app.js");
const ages=core.STANDARDS.who2000.rows.map(r=>r[0]);
const counts=[5,3,2,4,8,12,25,45,80,140,220,350,520,710,890,1050,1120,1200];
const result=core.calculate(ages.map((age,i)=>({age:age,count:counts[i],py:100000})),"who2000");
if(!Number.isFinite(result.asr_per_100k)||result.asr_per_100k<=0) throw new Error("ASR browser smoke test failed");
const only75=core.calculate(ages.map(age=>({age:age,count:age==="75-79"?100:0,py:100000})),"who2000");
if(only75.cumulative_rate_0_74_pct!==0) throw new Error("75-79 must not contribute to cumulative 0-74 rate");
console.log("Browser calculator smoke test passed");
