package br.com.rec_facial;


import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.os.Build;
import android.os.Bundle;
import android.text.InputType;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import br.com.rec_facial.camera.CameraActivity;
import br.com.rec_facial.recfacial.ReconhecimentoFacial;
import br.com.rec_facial.recfacial.SimilarityClassifier;

/**
 * @author Evalton.nunes
 * <p>
 * 20/12/2021
 */
public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    /******************RECONHECIMENTO FACIAL**************************/
    FaceDetector detector;
    Interpreter tfLite;
    private static String localFoto = "";

    boolean start = true,
            flipX = false;
    float[][] embeedings;
    private static int SELECT_PICTURE = 1;
    private static final int MY_CAMERA_REQUEST_CODE = 100;
    int[] intValues;
    int inputSize = 112;
    boolean isModelQuantized = false;
    float IMAGE_MEAN = 128.0f;
    float IMAGE_STD = 128.0f;
    int OUTPUT_SIZE = 192;
    int cam_face = CameraSelector.LENS_FACING_FRONT;

    String modelFile = "mobile_face_net.tflite";
    private HashMap<String, SimilarityClassifier.Recognition> registered = new HashMap<>();
    private ReconhecimentoFacial reconhecimentoFacial = new ReconhecimentoFacial();
    /******************RECONHECIMENTO FACIAL**************************/

    ImageView face_preview;

    TextView info_rec;
    Button iniciar, recFacial;
    Context context;


    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        context = this;
        localFoto = context.getExternalFilesDir(null).getAbsolutePath() + "/Foto/evalton.jpg";

        face_preview = findViewById(R.id.imageView);

        info_rec = findViewById(R.id.info_rec);

        recFacial = findViewById(R.id.reconhecimento_facial);
        recFacial.setVisibility(View.INVISIBLE);

        iniciar = findViewById(R.id.add_face);

        //Camera Permission
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
        }

        //CARREGA OS ROSTOS SALVOS
        SharedPreferences sharedPreferences = getSharedPreferences("HashMap", MODE_PRIVATE);

        try {
            tfLite = new Interpreter(reconhecimentoFacial.loadModelFile(MainActivity.this, modelFile));
        } catch (IOException e) {
            e.printStackTrace();
        }

        //Inicializa o detector de face
        FaceDetectorOptions highAccuracyOpts =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .build();
        detector = FaceDetection.getClient(highAccuracyOpts);

        capturarFotoDefault();
        registered = reconhecimentoFacial.readFromSP(sharedPreferences);
    }

    @Override
    public void onClick(View v) {

        switch (v.getId()) {
            case R.id.add_face:
                Intent addfaceIntent = new Intent(getApplicationContext(), CameraActivity.class);
                startActivityForResult(addfaceIntent, 2);
                break;
            case R.id.reconhecimento_facial:
                Intent recIntent = new Intent(getApplicationContext(), CameraActivity.class);
                startActivityForResult(recIntent, 3);
                break;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "permissão de camera concedida.", Toast.LENGTH_LONG).show();
            } else {
                Toast.makeText(this, "permissão de camera negada.", Toast.LENGTH_LONG).show();
            }
        }
    }

    public void recognizeImage(final Bitmap bitmap) {

        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4);
        imgData.order(ByteOrder.nativeOrder());
        intValues = new int[inputSize * inputSize];
        //get pixel values from Bitmap to normalize
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);

                }
            }
        }
        //imgData is input to our model
        Object[] inputArray = {imgData};

        Map<Integer, Object> outputMap = new HashMap<>();

        embeedings = new float[1][OUTPUT_SIZE];

        outputMap.put(0, embeedings);

        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);


        float distance = Float.MAX_VALUE;
        String id = "0";
        String label = "?";

        if(start == false){
            addFace();
        }else{
            if (registered.size() > 0) {

                final Pair<String, Float> nearest = findNearest(embeedings[0]);
                if (nearest != null) {
                    final String nome = nearest.first;
                    label = nome;
                    distance = nearest.second;
                    if (distance < 1.000f) {
                        if (info_rec != null) {
                            start = false;
                            info_rec.setText("Você é "+nome);
                            recFacial.setVisibility(View.INVISIBLE);
                        }
                    } else {

                        if (info_rec != null) {
                            start = false;
                            recFacial.setVisibility(View.INVISIBLE);
                            info_rec.setText("Face não reconhecida!");
                        }
                    }


                }
            }
        }

    }

    //Compare Faces by distance between face embeddings
    private Pair<String, Float> findNearest(float[] emb) {
        Pair<String, Float> ret = null;
        for (Map.Entry<String, SimilarityClassifier.Recognition> entry : registered.entrySet()) {
            if (entry.getValue().getExtra() != null) {
                final String name = entry.getKey();
                final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];
                float distance = 0;
                for (int i = 0; i < emb.length; i++) {
                    float diff = emb[i] - knownEmb[i];
                    distance += diff * diff;
                }
                distance = (float) Math.sqrt(distance);
                if (ret == null || distance < ret.second) {
                    ret = new Pair<>(name, distance);
                }
            }
        }
        return ret;
    }

    //adiciona uma face para detecção
    private void addFace() {
        {

            start = false;
            AlertDialog.Builder builder = new AlertDialog.Builder(context);
            builder.setTitle("Digite o Nome");

            final EditText input = new EditText(context);

            input.setInputType(InputType.TYPE_CLASS_TEXT);
            builder.setView(input);

            builder.setPositiveButton("Adicionar", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {

                    SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                            "0", "", -1f);
                    result.setExtra(embeedings);

                    registered.put(input.getText().toString(), result);
                    recFacial.setVisibility(View.VISIBLE);

                }
            });
            builder.setNegativeButton("Cancelar", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    dialog.cancel();
                }
            });

            builder.show();
        }
    }

    private void capturarFotoDefault() {
        try {
            reconhecimentoFacial.ajustarResolucaoImagem(localFoto);
            Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(), R.drawable.face_icon2);
            //  Bitmap bitmap = reconhecimentoFacial.rotateImage(BitmapFactory.decodeFile(localFoto),localFoto);
            InputImage impphoto = InputImage.fromBitmap(bitmap, 0);
            detector.process(impphoto).addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                @Override
                public void onSuccess(List<Face> faces) {

                    if (faces.size() != 0) {
                        Face face = faces.get(0);
                        System.out.println(face);
                        Bitmap frame_bmp = null;
                        frame_bmp = BitmapFactory.decodeResource(context.getResources(),
                                R.drawable.face_icon2);
                        Bitmap frame_bmp1 = bitmap;
                        RectF boundingBox = new RectF(face.getBoundingBox());
                        Bitmap cropped_face = reconhecimentoFacial.getCropBitmapByCPU(frame_bmp1, boundingBox);
                        Bitmap scaled = reconhecimentoFacial.getResizedBitmap(cropped_face, 112, 112);
                        recognizeImage(scaled);
                        face_preview.setImageBitmap(scaled);
                        try {
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }).addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception e) {
                    start = true;
                    Toast.makeText(context, "Failed to add", Toast.LENGTH_SHORT).show();
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if(requestCode == 2){
                start = false;
            }else{
                start = true;
            }

            try {
                Bitmap bitmap = (Bitmap) data.getExtras().get("foto");
                Bitmap watermarkimage = bitmap.copy(bitmap.getConfig(), true);

                InputImage impphoto = InputImage.fromBitmap(watermarkimage, 0);
                detector.process(impphoto).addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {

                        if (faces.size() != 0) {
                            face_preview.setVisibility(View.VISIBLE);
                            Face face = faces.get(0);
                            System.out.println(face);

                            Bitmap frame_bmp = watermarkimage;

                            Bitmap frame_bmp1 = reconhecimentoFacial.rotateBitmap(frame_bmp, 0, flipX, false);
                            RectF boundingBox = new RectF(face.getBoundingBox());
                            Bitmap cropped_face = reconhecimentoFacial.getCropBitmapByCPU(frame_bmp1, boundingBox);
                            Bitmap scaled = reconhecimentoFacial.getResizedBitmap(cropped_face, 112, 112);
                            recognizeImage(scaled);

                        }
                    }
                }).addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        start = true;
                        Toast.makeText(context, "Falha ao capturar a foto", Toast.LENGTH_SHORT).show();
                    }
                });
                if(requestCode == 2){
                    face_preview.setImageBitmap(bitmap);
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


}