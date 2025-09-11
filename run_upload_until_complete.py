#!/usr/bin/env python3

import subprocess
import time
import sys
import os

def run_upload_command():
    """Run the ./ridges.py upload command and return the output."""
    try:
        # Change to the directory containing ridges.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Run the command
        result = subprocess.run(
            ["./ridges.py", "upload"], 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 5 minutes")
        return -1, "", "Command timed out"
    except Exception as e:
        print(f"💥 Error running command: {e}")
        return -1, "", str(e)

def main():
    """Main function to run upload command until completion."""
    print("🚀 Starting upload retry script...")
    print("📝 Will run './ridges.py upload' until 'Upload Complete' appears")
    print("🛑 Press Ctrl+C to stop\n")
    
    attempt = 1
    max_attempts = 50000  # Safety limit
    
    while attempt <= max_attempts:
        print(f"🔄 Attempt {attempt}:")
        
        # Run the upload command
        return_code, stdout, stderr = run_upload_command()
        
        # Print the output
        if stdout:
            print("📤 Output:")
            print(stdout)
        if stderr:
            print("⚠️  Errors:")
            print(stderr)
        
        # Check if upload was successful
        if "Upload Complete" in stdout:
            print(f"✅ SUCCESS! Upload completed on attempt {attempt}")
            print("🎉 Script finished successfully!")
            sys.exit(0)
        
        # If command failed, show the return code
        if return_code != 0:
            print(f"❌ Command failed with return code: {return_code}")
        
        print(f"⏳ Upload not complete yet. Waiting 5 seconds before retry...\n")
        
        try:
            time.sleep(1)  # Wait 5 seconds between attempts
        except KeyboardInterrupt:
            print("\n🛑 Script interrupted by user")
            sys.exit(1)
        
        attempt += 1
    
    print(f"🚫 Maximum attempts ({max_attempts}) reached. Stopping.")
    sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user")
        sys.exit(1)
