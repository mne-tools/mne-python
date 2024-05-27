# Get the path to the user's USERPROFILE folder
$userProfilePath = [System.Environment]::GetFolderPath("UserProfile")

# Define the path to the new directory
$newDirectoryPath = "$userProfilePath\mne_data"

# Check if the directory already exists
if (Test-Path -Path $newDirectoryPath) {
    Write-Output "Found existing MNE data directory: $newDirectoryPath"
} else {
    # Create the directory since it does not exist
    try {
        New-Item -ItemType Directory -Path $newDirectoryPath -Force
        # Verify if the directory was created successfully
        if (Test-Path -Path $newDirectoryPath) {
            Write-Output "MNE data directory created: $newDirectoryPath"
        } else {
            Write-Output "Failed to create directory: $newDirectoryPath"
            exit 1
        }
    } catch {
        Write-Output "Error creating MNE data directory: $_"
        exit 1
    }
}
