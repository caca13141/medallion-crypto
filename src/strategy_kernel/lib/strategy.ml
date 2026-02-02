(* ------------------------------------------------------------------ *)
(* TYPE DEFINITIONS (Making illegal states unrepresentable)           *)
(* ------------------------------------------------------------------ *)

type position =
  | Long of float  (* Size *)
  | Short of float (* Size *)
  | Flat

type market_regime =
  | Normal
  | HighVol
  | Dislocated (* Spread > 50bps *)

type trend =
  | Bullish
  | Bearish
  | Sideways

type state = {
  position : position;
  pnl : float;
  regime : market_regime;
  inventory_limit : float;
}

type action =
  | BuyLimit of float * float (* price, size *)
  | SellLimit of float * float
  | MarketClose
  | NoAction

(* ------------------------------------------------------------------ *)
(* STRATEGIC EQUILIBRIUM KERNEL                                       *)
(* ------------------------------------------------------------------ *)

module EquilibriumKernel = struct
  type market_participant = 
    | LiquidityProvisioner
    | DirectionalFlow
    | AlphaEngine

  (** Calculates execution skew based on topological connectivity & latent activations *)
  let compute_execution_skew ~connectivity ~activation_score =
    (* connectivity represents manifold instability/liquidity dislocation *)
    let flow_risk = connectivity *. 2.5 in
    
    (* latent activation reflects high-capacity model directional confidence *)
    let model_signal = activation_score *. 1.2 in
    
    (* Optimize skew for risk-neutral execution against informed flow *)
    if flow_risk > 0.5 then
      model_signal -. (flow_risk *. 0.5) 
    else
      model_signal +. 0.1 
end

type market_data = {
  price : float;
  theo : float;
  imbalance : float;
  connectivity : float;      
  activation : float;  
}

let initial_state = {
  position = Flat;
  pnl = 0.0;
  regime = Normal;
  inventory_limit = 1.0; (* 1 BTC *)
}

(** State + MarketData -> Execution Action *)
let evaluate_policy state data =
  let skew = EquilibriumKernel.compute_execution_skew 
               ~connectivity:data.connectivity 
               ~activation_score:data.activation in
                    
  let divergence = (data.theo +. skew) -. data.price in
  
  match state.position with
  | Flat ->
      if divergence > 5.0 then 
        BuyLimit (data.price, 0.1)
      else if divergence < -5.0 then
        SellLimit (data.price, 0.1)
      else
        NoAction
        
  | Long size ->
      if divergence < -2.0 then 
        SellLimit (data.price, size) 
      else if size < state.inventory_limit && divergence > 15.0 then
        BuyLimit (data.price, 0.1)
      else
        NoAction

  | Short size ->
      if divergence > 2.0 then
        BuyLimit (data.price, size)
      else if size < state.inventory_limit && divergence < -15.0 then
        SellLimit (data.price, 0.1)
      else
        NoAction

(* ------------------------------------------------------------------ *)
(* RUNTIME SERVICE                                                    *)
(* ------------------------------------------------------------------ *)

let rec dispatch_loop state =
  Unix.sleepf 0.1; 
  
  let current_data = {
    price = 95000.0 +. (Random.float 10.0);
    theo = 95005.0; 
    imbalance = 0.5;
    connectivity = 0.42;
    activation = 0.8;
  } in

  let action = evaluate_policy state current_data in
  
  match action with
  | BuyLimit (p, s) -> 
      Printf.printf "[INFO] EXECUTION: BUY %.4f @ %.2f | SKEW_THEO: %.2f\n%!" s p (current_data.theo +. 0.5);
      dispatch_loop { state with position = Long (match state.position with Long sz -> sz +. s | _ -> s) } 
      
  | SellLimit (p, s) ->
      Printf.printf "[INFO] EXECUTION: SELL %.4f @ %.2f\n%!" s p;
      dispatch_loop { state with position = Short (match state.position with Short sz -> sz +. s | _ -> s) }

  | NoAction -> dispatch_loop state
  | _ -> dispatch_loop state

let run () =
  Random.self_init ();
  print_endline "[INFO] Strategy Kernel Online (Equilibrium Mode).";
  dispatch_loop initial_state
