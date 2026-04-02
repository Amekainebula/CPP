//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define ll long long
//#define ld long double
//#define ull unsigned long long
//#define endl '\n'
//using namespace std;
//int a[200005];
//void solve()
//{
//    int n;
//    int ne = 0;
//    bool flag = 1;
//    cin >> n;
//    for (int i = 1; i <= n; i++)
//        cin >> a[i];
//    for (int i = 1; i <= n; i++)
//    {
//        int x;
//        cin >> x;
//        a[i] = a[i] - x;
//        /*if (a[i] < 0 && ne < 0)flag = 0;
//        if (a[i] < 0)ne = a[i];
//        if (a[i] == 0 && ne < 0)
//            flag = 0;
//        if (a[i] >= 0 && flag)
//        {
//            int temp = -ne;
//            if (temp > a[i])
//                flag = 0;
//        }*/
//    }
//    sort(a + 1, a + n + 1);
//    if (a[1] >= 0)
//        flag = 1;
//    else
//    {
//        if (a[1] + a[2] < 0)
//            flag = 0;
//        else
//            flag = 1;
//    }
//    if (flag)
//    {
//        cout << "YES" << endl;
//    }
//    else
//    {
//        cout << "NO" << endl;
//    }
//}
//int main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    ll T;
//    cin >> T;
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}