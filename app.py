"""
Main application for hoodie classification.
Provides both CLI and FastAPI interfaces.
"""

import argparse
import json
import time
import os

from typing import Dict, Any

from classifier import HoodieClassifier

def classify_single_image(image_path: str, config_path: str = "config.yaml", provider_override: str = None) -> Dict[str, Any]:
    """Classify a single hoodie image."""
    print("Loading model for single image classification...")
    
    # Load config and apply provider override if specified
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if provider_override:
        config['model_provider'] = provider_override
        print(f"Provider overridden to: {provider_override}")
    
    classifier = HoodieClassifier(config)
    
    start_time = time.time()
    result = classifier.classify(image_path)
    inference_time = time.time() - start_time
    
    result["inference_time"] = inference_time
    result["image_path"] = image_path
    
    return result

def classify_batch(images_dir: str, output_file: str = "predictions.csv", config_path: str = "config.yaml"):
    """Classify multiple images in a directory."""
    import csv
    from glob import glob
    
    print("Loading model and reference embeddings...")
    classifier = HoodieClassifier(config_path)
    print("Model loaded successfully - processing images...")
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(images_dir, ext)))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Classify each image using the SAME classifier instance
    results = []
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_path}")
        try:
            start_time = time.time()
            result = classifier.classify(img_path)
            inference_time = time.time() - start_time
            
            result["inference_time"] = inference_time
            result["image_path"] = img_path
            results.append(result)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({
                "image_path": img_path,
                "error": str(e),
                "pieces": None,
                "scores": {"2": None, "3": None},
                "margin": None,
                "fallback_used": None,
                "inference_time": None
            })
    
    # Save results
    if output_file.endswith('.csv'):
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'image_path', 'pieces', 'score_2', 'score_3', 'margin', 
                'fallback_used', 'inference_time', 'error'
            ])
            writer.writeheader()
            
            for result in results:
                row = {
                    'image_path': result['image_path'],
                    'pieces': result.get('pieces'),
                    'score_2': result.get('scores', {}).get('2'),
                    'score_3': result.get('scores', {}).get('3'),
                    'margin': result.get('margin'),
                    'fallback_used': result.get('fallback_used'),
                    'inference_time': result.get('inference_time'),
                    'error': result.get('error')
                }
                writer.writerow(row)
    else:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def generate_reference_embeddings(images_dir: str, config_path: str = "config.yaml"):
    """Generate reference embeddings from the images directory."""
    classifier = HoodieClassifier(config_path)
    classifier.generate_reference_embeddings(images_dir)

def analyze_edge_cases(edge_case_dir: str = "edge_case_images", config_path: str = "config.yaml", provider_override: str = None):
    """Analyze edge case images to test system robustness."""
    import csv
    from glob import glob
    from datetime import datetime
    
    print(f"Analyzing edge cases in: {edge_case_dir}")
    
    # Load config and apply provider override if specified
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if provider_override:
        config['model_provider'] = provider_override
        print(f"Provider overridden to: {provider_override}")
    
    # Initialize classifier
    classifier = HoodieClassifier(config)
    print("Classifier initialized for edge case analysis")
    
    # Find all edge case images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    edge_case_files = []
    for ext in image_extensions:
        edge_case_files.extend(glob(os.path.join(edge_case_dir, ext)))
    
    if not edge_case_files:
        print(f"No edge case images found in {edge_case_dir}")
        return
    
    print(f"Found {len(edge_case_files)} edge case images")
    
    # Analyze each edge case
    results = []
    for i, img_path in enumerate(edge_case_files):
        print(f"\nProcessing edge case {i+1}/{len(edge_case_files)}: {os.path.basename(img_path)}")
        
        try:
            start_time = time.time()
            result = classifier.classify(img_path)
            inference_time = time.time() - start_time
            
            # Add edge case specific metadata
            result["image_path"] = img_path
            result["image_name"] = os.path.basename(img_path)
            result["inference_time"] = inference_time
            result["edge_case_type"] = _identify_edge_case_type(img_path)
            result["analysis_timestamp"] = datetime.now().isoformat()
            
            results.append(result)
            
            # Print detailed analysis
            print(f"   Classification: {result['pieces']}-piece")
            print(f"   Confidence: 2-piece={result['scores']['2']:.3f}, 3-piece={result['scores']['3']:.3f}")
            print(f"   Margin: {result['margin']:.3f}")
            print(f"   Fallback used: {result['fallback_used']}")
            print(f"   Time: {inference_time:.3f}s")
            print(f"   Edge case type: {result['edge_case_type']}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({
                "image_path": img_path,
                "image_name": os.path.basename(img_path),
                "error": str(e),
                "pieces": None,
                "scores": {"2": None, "3": None},
                "margin": None,
                "fallback_used": None,
                "inference_time": None,
                "edge_case_type": "error",
                "analysis_timestamp": datetime.now().isoformat()
            })
    
    # Generate timestamped output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"edge_case_analysis_{timestamp}.csv"
    json_filename = f"edge_case_analysis_{timestamp}.json"
    
    # Save CSV results
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_name', 'pieces', 'score_2', 'score_3', 'margin', 
            'fallback_used', 'inference_time', 'edge_case_type', 'error'
        ])
        writer.writeheader()
        
        for result in results:
            row = {
                'image_name': result.get('image_name'),
                'pieces': result.get('pieces'),
                'score_2': result.get('scores', {}).get('2'),
                'score_3': result.get('scores', {}).get('3'),
                'margin': result.get('margin'),
                'fallback_used': result.get('fallback_used'),
                'inference_time': result.get('inference_time'),
                'edge_case_type': result.get('edge_case_type'),
                'error': result.get('error')
            }
            writer.writerow(row)
    
    # Save detailed JSON results
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEDGE CASE ANALYSIS COMPLETE!")
    print(f"   CSV Results: {csv_filename}")
    print(f"   Detailed Results: {json_filename}")
    print(f"   Processed {len(edge_case_files)} edge cases")
    
    return results

def _identify_edge_case_type(image_path: str) -> str:
    """Identify the type of edge case based on filename or path."""
    filename = os.path.basename(image_path).lower()
    
    if 'edge_case' in filename:
        return "explicit_edge_case"
    elif 'worn' in filename or 'body' in filename or 'person' in filename:
        return "worn_hoodie"
    elif 'complex' in filename or 'background' in filename:
        return "complex_background"
    elif 'partial' in filename or 'obscured' in filename:
        return "partial_visibility"
    elif 'lighting' in filename or 'shadow' in filename:
        return "lighting_variation"
    else:
        return "unknown_edge_case"

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Hoodie Piece Classifier")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Classify single image
    classify_parser = subparsers.add_parser('classify', help='Classify a single image')
    classify_parser.add_argument('--path', required=True, help='Path to image file')
    classify_parser.add_argument('--provider', choices=['local', 'openai', 'huggingface-api'], 
                               help='Override model provider (local, openai, or huggingface-api)')
    classify_parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    classify_parser.add_argument('--output', help='Output file for results (JSON)')
    
    # Batch classification
    batch_parser = subparsers.add_parser('classify-batch', help='Classify multiple images')
    batch_parser.add_argument('--dir', required=True, help='Directory containing images')
    batch_parser.add_argument('--provider', choices=['local', 'openai', 'huggingface-api'], 
                             help='Override model provider (local, openai, or huggingface-api)')
    batch_parser.add_argument('--out', default='predictions.csv', help='Output file path')
    batch_parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    
    # Generate reference embeddings
    ref_parser = subparsers.add_parser('generate-refs', help='Generate reference embeddings')
    ref_parser.add_argument('--dir', required=True, help='Directory containing reference images')
    ref_parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    

    
    # Edge case analysis
    edge_parser = subparsers.add_parser('edge-cases', help='Analyze edge case images for robustness testing')
    edge_parser.add_argument('--dir', default='edge_case_images', help='Directory containing edge case images')
    edge_parser.add_argument('--provider', choices=['local', 'openai', 'huggingface-api'], 
                           help='Override model provider (local, openai, or huggingface-api)')
    edge_parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    
    # API server
    api_parser = subparsers.add_parser('api', help='Start API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    api_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    api_parser.add_argument('--provider', choices=['local', 'openai', 'huggingface-api'], 
                           help='Override model provider (local, openai, or huggingface-api)')
    
    args = parser.parse_args()
    
    if args.command == 'classify':
        result = classify_single_image(args.path, args.config, args.provider)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    elif args.command == 'classify-batch':
        classify_batch(args.dir, args.out, args.config)
    
    elif args.command == 'generate-refs':
        generate_reference_embeddings(args.dir, args.config)
    
    elif args.command == 'edge-cases':
        analyze_edge_cases(args.dir, args.config, args.provider)
    
    elif args.command == 'api':
        try:
            from fastapi import FastAPI, File, UploadFile
            from fastapi.responses import HTMLResponse, JSONResponse
            from fastapi.middleware.cors import CORSMiddleware
            import uvicorn
            
            app = FastAPI(title="Hoodie Classifier API", version="1.0.0")
            
            # Add CORS middleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Allows all origins
                allow_credentials=True,
                allow_methods=["*"],  # Allows all methods
                allow_headers=["*"],  # Allows all headers
            )
            
            # Create a single classifier instance for reuse
            classifier = None
            
            @app.get("/", response_class=HTMLResponse)
            async def root():
                """Serve the web interface."""
                with open('index.html', 'r') as f:
                    html_content = f.read()
                return html_content
            
            @app.post("/classify")
            async def classify_image(file: UploadFile = File(...)):
                """Classify a hoodie image via API."""
                nonlocal classifier
                
                try:
                    # Initialize classifier once if not exists
                    if classifier is None:
                        import yaml
                        
                        with open('config.yaml', 'r') as f:
                            config = yaml.safe_load(f)
                        
                        # Apply provider override if specified
                        if args.provider:
                            config['model_provider'] = args.provider
                            print(f"API Provider overridden to: {args.provider}")
                        
                        classifier = HoodieClassifier(config)
                        print(f"Initialized classifier with config: {config}")
                    
                    # Save uploaded file temporarily
                    temp_path = f"temp_{file.filename}"
                    with open(temp_path, "wb") as f:
                        content = await file.read()
                        f.write(content)
                    
                    print(f"Classifying image: {temp_path}")
                    print(f"Classifier type: {type(classifier)}")
                    print(f"Classifier methods: {[m for m in dir(classifier) if not m.startswith('_')]}")
                    
                    # Classify using the reusable classifier
                    result = classifier.classify(temp_path)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    return JSONResponse(content=result)
                
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Error in classify_image: {e}")
                    print(f"Error details: {error_details}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": str(e), "details": error_details}
                    )
            
            @app.get("/health")
            async def health_check():
                """Health check endpoint."""
                return {"status": "healthy"}
            
            print(f"Starting API server on {args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port)
            
        except ImportError:
            print("Error: FastAPI not installed. Run: pip install fastapi uvicorn")
            return
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()


