product_version = "4.15"
documents = [
    "about",
    "getting_started",
    "release_notes",
    "security_and_compliance",
    "architecture",
    "support",
    "installing",
    "installing_openShift_container_platform_with_the_assisted_installer",
    "updating_clusters",
    "authentication_and_authorization",
    "networking",
    "registry",
    "postinstallation_configuration",
    "storage",
    "scalability_and_performance",
    "edge_computing",
    "migrating_from_version_3_to_4",
    "Migration_Toolkit_for_Containers",
    "backup_and_restore",
    "machine_management",
    "web_console",
    "hosted_control_planes",
    "cli_tools",
    "buildings_applications",
    "serverless",
    "images",
    "nodes",
    "operators",
    "specialized_hardware_and_driver_enablement",
    "openShift_sandboxed_containers_release_notes",
    "openShift_sandboxed_containers_user_guide",
    "Builds_using_Shipwright",
    "Builds_using_BuildConfig",
    "gitops",
    "pipelines",
    "jenkins",
    "monitoring",
    "logging",
    "distributed_tracing",
    "red_hat_build_of_opentelemetry",
    "network_observability",
    "power_monitoring",
    "cluster_observability_operator",
    "virtualization",
    "service_mesh",
    "Windows_Container_Support_for_OpenShift"  
]

pdfs_ocp = [f"https://access.redhat.com/documentation/de-de/openshift_container_platform/{product_version}/pdf/{doc}/OpenShift_Container_Platform-{product_version}-{doc}-en-us.pdf" for doc in documents]
pdfs_to_urls_ocp = {f"openshift_container_platform-{product_version}-{doc}-en-us": f"https://access.redhat.com/documentation/de-de/openshift_container_platform/{product_version}/html-single/{doc}/index" for doc in documents}



os.mkdir(f"ocp-doc-{product_version}")

for pdf in pdfs_ocp:
    try:
        response = requests.get(pdf)
    except:
        print(f"Skipped {pdf}")
        continue
    if response.status_code!=200:
        print(f"Skipped {pdf}")
        continue  
    with open(f"ocp-doc-{product_version}/{pdf.split('/')[-1]}", 'wb') as f:
        f.write(response.content)


pdf_folder_path = f"./ocp-doc-{product_version}"

pdf_loader = PyPDFDirectoryLoader(pdf_folder_path)
pdf_docs = pdf_loader.load()