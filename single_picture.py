import cv2, numpy as np
import hailo_platform as hpf

# Load the compiled Hailo model (YOLOv8 HEF file)
hef_path = "yolov8m.hef"
hef = hpf.HEF(hef_path)

# Configure the Hailo device (Virtual Device) with the HEF
with hpf.VDevice() as device:
    cfg = hpf.ConfigureParams.create_from_hef(
        hef, interface=hpf.HailoStreamInterface.PCIe
    )
    network_group = device.configure(hef, cfg)[0]
    network_params = network_group.create_params()

    # Get input/output stream info from the HEF (assuming single input network)
    input_info = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[
        0
    ]  # YOLOv8 with NMS gives one output stream

    # Prepare input/output vstreams with proper format
    input_vstreams = hpf.InputVStreamParams.make_from_network_group(
        network_group, quantized=False, format_type=hpf.FormatType.UINT8
    )
    output_vstreams = hpf.OutputVStreamParams.make_from_network_group(
        network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
    )

    # Read and preprocess an image frame
    frame = cv2.imread("zzz.jpg")
    H, W = (
        input_info.shape[0],
        input_info.shape[1],
    )  # Expected height and width (NHWC order)
    frame_resized = cv2.resize(frame, (W, H))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    input_array = np.expand_dims(frame_rgb, axis=0).astype(
        np.uint8
    )  # shape: (1, H, W, 3)

    # Run inference on the Hailo-8
    with network_group.activate(network_params):
        with hpf.InferVStreams(
            network_group, input_vstreams, output_vstreams
        ) as pipeline:
            results = pipeline.infer({input_info.name: input_array})

    for idx, class_detections in enumerate(results[list(results.keys())[0]][0]):
        if class_detections.shape[0] > 0:
            for det in class_detections:
                print(idx)
                print(det)
                print(det[0:4] * 640)
