def ace_score(PINC, electricity, cold, heat, y_test):
    count_elec = 0
    count_cold = 0
    count_heat = 0
    
    if (100-int(PINC*100)) % 2 == 0:
        lb = int((100-int(PINC*100))/2)-1
    else:
        lb = int((100-int(confidence_level*100))/2)
    ub = -((100-int(PINC*100))-lb-1)
    
    for i in range(electricity.size(-1)):
        if electricity[lb, i] <= y_test.cpu()[i, 0] <= electricity[ub, i]:
            count_elec += 1
        if cold[lb, i] <= y_test.cpu()[i, 1] <= cold[ub, i]:
            count_cold += 1
        if heat[lb, i] <= y_test.cpu()[i, 2] <= heat[ub, i]:
            count_heat += 1
    return count_elec/electricity.size(-1)-PINC, count_cold/electricity.size(-1)-PINC, count_heat/electricity.size(-1)-PINC


def sharpness_score(PINC, electricity, cold, heat, y_test):
    sharpness_elec = 0
    sharpness_cold = 0
    sharpness_heat = 0
    total_width_elec = 0
    total_width_cold = 0
    total_width_heat = 0
    
    if (100-int(PINC*100)) % 2 == 0:
        lb = int((100-int(PINC*100))/2)-1
    else:
        lb = int((100-int(confidence_level*100))/2)
    ub = -((100-int(PINC*100))-lb-1)
    
    for i in range(electricity.size(-1)):
        width_elec = electricity[ub, i]-electricity[lb, i]
        width_cold = cold[ub, i]-cold[lb, i]
        width_heat = heat[ub, i]-heat[lb, i]
        total_width_elec += width_elec
        total_width_cold += width_cold
        total_width_heat += width_heat
        
        if electricity[lb, i] <= y_test.cpu()[i, 0] <= electricity[ub, i]:
            sharpness_elec+=-2*(1-PINC)*width_elec
        elif y_test.cpu()[i, 0]< electricity[lb, i]:
            sharpness_elec = sharpness_elec-2*(1-PINC)*width_elec-4*(electricity[lb, i]-y_test.cpu()[i, 0])
        else:
            sharpness_elec = sharpness_elec-2*(1-PINC)*width_elec-4*(y_test.cpu()[i, 0]-electricity[ub, i])
            
        if cold[lb, i] <= y_test.cpu()[i, 1] <= cold[ub, i]:
            sharpness_cold+=-2*(1-PINC)*width_cold
        elif y_test.cpu()[i, 1]< cold[lb, i]:
            sharpness_cold = sharpness_cold-2*(1-PINC)*width_cold-4*(cold[lb, i]-y_test.cpu()[i, 1])
        else:
            sharpness_cold = sharpness_cold-2*(1-PINC)*width_cold-4*(y_test.cpu()[i, 1]-cold[ub, i])
        
        if heat[lb, i] <= y_test.cpu()[i, 2] <= heat[ub, i]:
            sharpness_heat+=-2*(1-PINC)*width_heat
        elif y_test.cpu()[i, 2]< heat[lb, i]:
            sharpness_heat = sharpness_heat-2*(1-PINC)*width_heat-4*(heat[lb, i]-y_test.cpu()[i, 2])
        else:
            sharpness_heat = sharpness_heat-2*(1-PINC)*width_heat-4*(y_test.cpu()[i, 2]-heat[ub, i])
       
    
    return sharpness_elec/electricity.size(-1), sharpness_cold/electricity.size(-1), sharpness_heat/electricity.size(-1), total_width_elec/electricity.size(-1), total_width_cold/electricity.size(-1), total_width_heat/electricity.size(-1)