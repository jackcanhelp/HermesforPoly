try:
    from reflection_engine import run_reflection_cycle
    from main import main
    print("Test: Running reflection cycle...")
    run_reflection_cycle()
    print("Test: Reflection cycle OK. Running main()...")
    main()
    print("Test: main() OK. System is functional!")
except Exception as e:
    import traceback
    traceback.print_exc()
