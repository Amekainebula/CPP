//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define int long long
//#define double long double
//#define ull unsigned long long
//#define endl '\n'
//using namespace std;
//int t;
//signed main()
//{
//    cin >> t;
//    while (t--)
//    {
//        int n, a, b, c;
//        int sum = 0;
//        int ans = 0;
//        cin >> n >> a >> b >> c;
//        ans = n / (a + b + c) + 1;
//        sum += ans * (a + b + c);
//        ans *= 3;
//        if (sum > n)
//        {
//            sum -= c;
//            ans--;
//        }
//        if (sum < n)
//        {
//            ans++;
//            cout << ans << endl;
//            continue;
//        }
//        if (sum > n)
//        {
//            sum -= b;
//            ans--;
//        }
//        if (sum < n)
//        {
//            ans++;
//            cout << ans << endl;
//            continue;
//        }
//        if (sum > n)
//        {
//            sum -= a;
//            ans--;
//        }
//        if (sum < n)
//        {
//            ans++;
//            cout << ans << endl;
//            continue;
//        }
//        cout << ans << endl;
//    }
//    return 0;
//}