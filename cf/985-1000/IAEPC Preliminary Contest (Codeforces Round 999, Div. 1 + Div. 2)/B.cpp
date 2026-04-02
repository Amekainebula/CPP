//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define int long long
//#define ld long double
//#define ull unsigned long long
//#define lowbit(x) (x & -x)
//#define pb push_back
//#define pii pair<int, int>
//#define mp make_pair
//#define fi first
//#define se second
//#define all(x) x.begin(), x.end()
//#define rall(x) x.rbegin(), x.rend()
//#define sz(x) (int)(x).size()
//#define endl '\n'
//using namespace std;
//int a[200005];
//int arr[200005];
//int arr2[200005];
//void solve()
//{
//    map<int, int> mpp;
//    int n;
//    int ans = -1;
//    int flag = -1;
//    int cntt = 0;
//    //int maxx = -1;
//    //int minn = 1e9;
//    cin >> n;
//    for (int i = 1; i <= n; i++)
//    {
//        int x;
//        cin >> x;
//        //minn = min(minn, x);
//        a[i] = x;
//        mpp[x]++;
//        if (mpp[x] == 2)
//            arr2[++cntt] = x;
//        if (mpp[x] == 4)
//        {
//            ans = x;
//            flag = 1;
//        }
//        if (mpp[x] >= 2 && flag == -1)
//        {
//            flag = 0;
//            //maxx = max(maxx, x);
//        }
//    }
//    if (flag == 1)
//    {
//        for (int i = 1; i <= 4; i++)
//            cout << ans << " ";
//        cout << endl;
//        return;
//    }
//    else if (flag == -1)
//    {
//        cout << "-1" << endl;
//        return;
//    }
//    if (cntt >= 2)
//    {
//        cout << arr2[1] << " " << arr2[1] << " " << arr2[2] << " " << arr2[2] << " " << endl;
//        return;
//    }
//    sort(a + 1, a + n + 1);
//    /*arr[1] = -1;
//    for (int i = 2; i <= n; i++)
//    {
//        arr[i] = a[i] - a[i - 1];
//    }*/
//    if (a[n - 1] == a[n])
//    {
//        cout << a[n - 1] << " " << a[n] << " " << a[n - 2] << " " << a[n - 3] << " " << endl;
//        return;
//    }
//    if (mpp[arr2[1]] == 3)
//    {
//        for (int i = 1; i <= n; i++)
//        {
//            if (a[i] == arr2[1])continue;
//            if (a[i] < arr2[1] * 3)
//            {
//                cout << a[i] << " " << arr2[1] << " " << arr2[1] << " " << arr2[1] << " " << endl;
//                return;
//            }
//            else
//            {
//                cout << "-1" << endl;
//                return;
//            }
//        }
//    }
//    else
//    {
//        for (int i = 1; i <= n; i++)
//        {
//            if (a[i] == arr2[1])continue;
//            int temp = a[i] + arr2[1] * 2;
//            for (int j = i + 1; j <= n; j++)
//            {
//                if (a[j] == arr2[1])continue;
//                if (temp > a[j])
//                {
//                    cout << a[i] << " " << arr2[1] << " " << arr2[1] << " " << a[j] << " " << endl;
//                    return;
//                }
//                else
//                    break;
//            }
//        }
//    }
//    cout << "-1" << endl;
//}
//signed main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    cout.tie(0);
//    int T = 1;
//    cin >> T;
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}