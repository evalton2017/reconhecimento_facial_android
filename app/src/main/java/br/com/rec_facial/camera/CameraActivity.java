package br.com.rec_facial.camera;


import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import br.com.rec_facial.R;
import br.com.rec_facial.recfacial.ReconhecimentoFacial;


public class CameraActivity extends AppCompatActivity implements View.OnClickListener  {

    private PreviewView previewView;
    private CameraSelector cameraSelector;
    private ReconhecimentoFacial reconhecimentoFacial = new ReconhecimentoFacial();
    FaceDetector detector;
    String modelFile = "mobile_face_net.tflite";

    boolean start = true,
            flipX = false;
    float[][] embeedings;
    private static int SELECT_PICTURE = 1;
    ProcessCameraProvider cameraProvider;
    private static final int MY_CAMERA_REQUEST_CODE = 100;
    int[] intValues;
    int inputSize = 112;
    boolean isModelQuantized = false;
    float IMAGE_MEAN = 128.0f;
    float IMAGE_STD = 128.0f;
    int OUTPUT_SIZE = 192;
    int cam_face = CameraSelector.LENS_FACING_FRONT;
    private ImageButton captura;
    private Context context = this;
    private Bitmap bitmap = null;
    private final int REQUEST_CODE = 100;

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    private String localFoto = "";
    ImageView face_preview;

    @RequiresApi(api = Build.VERSION_CODES.M)
    @SuppressLint("WrongViewCast")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        captura = findViewById(R.id.captureButton);
        previewView = findViewById(R.id.previewView);
        face_preview = findViewById(R.id.imageView_rec);

        //Camera Permission
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
        }

        //Inicializa o detector de face
        FaceDetectorOptions highAccuracyOpts =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .build();
        detector = FaceDetection.getClient(highAccuracyOpts);

        cameraBind();

        captura.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = getIntent();
                i.putExtra("foto", bitmap);
                setResult(RESULT_OK, i);
                finish();

            }
        });

    }

    private void cameraBind() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        previewView = findViewById(R.id.previewView);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {

            }
        }, ContextCompat.getMainExecutor(this));
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {

        Preview preview = new Preview.Builder()
                .build();

        cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(cam_face)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        Executor executor = Executors.newSingleThreadExecutor();
        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy imageProxy) {
                InputImage image = null;
                @SuppressLint({"UnsafeExperimentalUsageError", "UnsafeOptInUsageError"})
                Image mediaImage = imageProxy.getImage();

                if (mediaImage != null) {
                    image = InputImage.fromMediaImage(mediaImage, imageProxy.getImageInfo().getRotationDegrees());
                    System.out.println("Rotation " + imageProxy.getImageInfo().getRotationDegrees());
                }
                System.out.println("ANALYSIS");
                Task<List<Face>> result =
                        detector.process(image)
                                .addOnSuccessListener(
                                        new OnSuccessListener<List<Face>>() {
                                            @Override
                                            public void onSuccess(List<Face> faces) {

                                                if (faces.size() != 0) {

                                                    Face face = faces.get(0);

                                                    Bitmap frame_bmp = reconhecimentoFacial.toBitmap(mediaImage);

                                                    int rot = imageProxy.getImageInfo().getRotationDegrees();

                                                    Bitmap frame_bmp1 = reconhecimentoFacial.rotateBitmap(frame_bmp, rot, false, false);

                                                    RectF boundingBox = new RectF(face.getBoundingBox());

                                                    Bitmap cropped_face = reconhecimentoFacial.getCropBitmapByCPU(frame_bmp1, boundingBox);

                                                    if (flipX)
                                                        cropped_face = reconhecimentoFacial.rotateBitmap(cropped_face, 0, flipX, false);
                                                    bitmap = reconhecimentoFacial.getResizedBitmap(cropped_face, 112, 112);
                                                    face_preview.setImageBitmap(bitmap);
                                                    try {
                                                        Thread.sleep(10);
                                                    } catch (InterruptedException e) {
                                                        e.printStackTrace();
                                                    }
                                                }

                                            }
                                        })
                                .addOnFailureListener(
                                        new OnFailureListener() {
                                            @Override
                                            public void onFailure(@NonNull Exception e) {
                                                Toast.makeText(getApplicationContext(), "Erro ao capturar foto", '0').show();
                                            }
                                        })
                                .addOnCompleteListener(new OnCompleteListener<List<Face>>() {
                                    @Override
                                    public void onComplete(@NonNull Task<List<Face>> task) {
                                        imageProxy.close();
                                    }
                                });
            }
        });

        cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, imageAnalysis, preview);

    }

    private  String getCurSysDate(){
        return new SimpleDateFormat("yyy-MM-dd_HH-mm-ss").format(new Date());
    }

    public Boolean gravarFotos(Bitmap bitmap) {
        String filename = "capture_" + getCurSysDate() + ".png";
        Boolean retorno = true;
        File file = null;
        File defaultFile = new File(context.getExternalFilesDir(null).getAbsolutePath() + "/Fotos");
        if (!defaultFile.exists()) {
            defaultFile.mkdir();
        }
        file = new File(defaultFile, filename);
        localFoto = file.getAbsolutePath();
        if (file != null) {
            file.delete();
            file = new File(defaultFile, filename);
        }
        FileOutputStream fos;
        try {
            fos = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 80, fos);
            fos.flush();
            fos.close();
        } catch (FileNotFoundException e) {
            Log.e("Filenotfound", e.getMessage(), e);
            retorno = false;
        } catch (IOException e) {
            Log.e("IOException", e.getMessage(), e);
            retorno = false;
        }
        return retorno;
    }

    @Override
    public void onResume(){
        super.onResume();
    }

    @Override
    public void onPause(){
        super.onPause();
    }

    @Override
    public void onStart(){
        super.onStart();
    }

    @Override
    public void onDestroy(){
        super.onDestroy();
    }

    @Override
    public void onClick(View view) {

    }
}